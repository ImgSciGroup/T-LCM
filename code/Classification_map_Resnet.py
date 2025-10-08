import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import linecache
import tifffile
import cv2
def extract_patch_and_average(padded_image, coord, patch_size):
    half_size = patch_size // 2
    x, y = coord
    x += half_size  # 调整坐标以匹配填充后的图像
    y += half_size
    patch = padded_image[x - half_size:x + half_size + 1, y - half_size:y + half_size + 1]
    return patch
def Get_intrinal_sample(Path):
    Init_label = []
    Init_coor = []
    for j in range(1, len(open(Path).readlines()) + 1):
        lindata = linecache.getline(Path, j)
        data_label,data_x, data_y = int(lindata.split(',')[0]), int(lindata.split(',')[1]), int(lindata.split(',')[2])
        Init_coor.append([data_x, data_y])
        Init_label.append(data_label)
    return Init_coor,Init_label
# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HyperspectralDataset(Dataset):
    def __init__(self, patches, labels):
        self.patches = patches
        self.labels = labels

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        label = self.labels[idx]
        return torch.tensor(patch, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=6):
        super(ResNet, self).__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv2d(10, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 使用自适应平均池化层
        self.linear = nn.Linear(256 * 1 * 1, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)  # 使用自适应平均池化层
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])




Time = ['201812', '201901', '201902', '201903', '201904', '201905', '201906', '201907', '201908', '201909', '201910',
        '201911', '201912',
        '202001', '202002', '202003', '202004', '202005', '202006', '202007', '202008', '202009', '202010',
        '202011', '202012',
        '202101', '202102', '202103', '202104', '202105', '202106', '202107', '202108', '202109', '202110',
        '202111', '202112',
        '202201', '202202', '202203', '202204', '202205', '202206', '202207', '202208', '202209', '202210',
        '202211']
img_path = '../Data/WuHan'
path = '../Data/WuHan\Intermediate results\SVM\\120\\4000'
patch = 7
for date in Time:
    Init_coor, Init_label = Get_intrinal_sample(os.path.join(path, "{0}/WHU_{1}_initial.txt".format(date, date)))
    coords = np.array(Init_coor)
    labels = np.array(Init_label)

    spectral_image = tifffile.imread(os.path.join(img_path, "{}.tif".format(date)))
    padding = int(patch / 2)  # 对应于patch的一半尺寸（5x5的patch则padding为2）
    padded_image = np.pad(spectral_image, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')

    # 提取所有坐标的patch并平均
    patches = np.array([extract_patch_and_average(padded_image, coord, patch) for coord in coords])

    # 转换为Dataset
    train_dataset = HyperspectralDataset(patches, labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


    # 初始化模型、损失函数和优化器
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.permute(0, 3, 1, 2))
            loss = criterion(outputs, labels - 1)  # 标签减1，因为标签从1开始
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")


    model.eval()
    classification_map = np.zeros((spectral_image.shape[0], spectral_image.shape[1]))
    with torch.no_grad():
        for i in range(spectral_image.shape[0]):
            for j in range(spectral_image.shape[1]):
                patch = padded_image[i:i+7, j:j+7, :]
                patch = torch.tensor(patch, dtype=torch.float32).to(device).unsqueeze(0).permute(0, 3, 1, 2)
                output = model(patch)
                _, predicted = torch.max(output.data, 1)
                classification_map[i, j] = predicted.item() + 1  # 标签加1，因为预测值从0开始

    cv2.imwrite(os.path.join(img_path, "Result/{0}_{1}_result.bmp".format(date, 'Resnet')), classification_map)







