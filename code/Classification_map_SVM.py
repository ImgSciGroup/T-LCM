import os.path
import numpy as np
import cv2
from sklearn import svm
import matplotlib.pyplot as plt
import cv2
import tifffile
import linecache
def Classification(image, coordinates, labels, patchsize, id, path):

    # 提取5x5x10 patch并将其平均为1x10向量
    def extract_patch_and_average(padded_image, coord, patch_size):
        half_size = patch_size // 2
        x, y = coord
        x += padding  # 调整坐标以匹配填充后的图像
        y += padding
        patch = padded_image[x - half_size:x + half_size + 1, y - half_size:y + half_size + 1]
        return np.mean(patch, axis=(0, 1))



    # 对整个影像进行5x5 patch提取和分类预测
    def classify_image(padded_image, clf, patch_size):
        height, width, channels = padded_image.shape
        height -= 2 * padding
        width -= 2 * padding
        classification_map = np.zeros((height, width), dtype=int)

        for i in range(height):
            for j in range(width):
                patch_vector = extract_patch_and_average(padded_image, (i, j), patch_size)
                classification_map[i, j] = clf.predict([patch_vector])[0]

        return classification_map


    # 添加填充以确保提取patch时不会超出图像范围
    padding = int(patchsize/2)  # 对应于patch的一半尺寸（5x5的patch则padding为2）
    padded_image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')

    # 提取所有坐标的patch并平均
    features = np.array([extract_patch_and_average(padded_image, coord, patchsize) for coord in coordinates])

    # 使用提取的向量和标签训练SVM
    clf = svm.SVC(kernel='rbf', C=10.0)
    clf.fit(features, labels)
    classification_map = classify_image(padded_image, clf, patchsize)
    cv2.imwrite(os.path.join(path,"Result/{0}_result.bmp".format(id)), classification_map)

def Get_intrinal_sample(Path):
    Init_label = []
    Init_coor = []
    for j in range(1, len(open(Path).readlines()) + 1):
        lindata = linecache.getline(Path, j)
        data_label,data_x, data_y = int(lindata.split(',')[0]), int(lindata.split(',')[1]), int(lindata.split(',')[2])
        Init_coor.append([data_x, data_y])
        Init_label.append(data_label)
    return Init_coor,Init_label

if __name__ == '__main__':
    Time = ['201812','201901', '201902', '201903', '201904', '201905', '201906', '201907', '201908', '201909', '201910','201911', '201912',
            '202001', '202002', '202003', '202004', '202005', '202006', '202007', '202008', '202009', '202010', '202011', '202012',
            '202101', '202102', '202103', '202104', '202105', '202106', '202107', '202108', '202109', '202110', '202111', '202112',
            '202201', '202202', '202203', '202204', '202205', '202206', '202207', '202208', '202209', '202210']
    img_path = '../Data/WuHan'
    path = '../Data/WuHan\Intermediate results\SVM\\30\\4000'
    for date in Time:
        img_mat = tifffile.imread(os.path.join(img_path,"{}.tif".format(date)))
        patch = 7
        #initial
        Init_coor, Init_label = Get_intrinal_sample(os.path.join(path,"{0}\WHU_{1}_initial.txt".format(date,date)))
        selected_coordinates = np.array(Init_coor)
        selected_labels = np.array(Init_label)
        Classification(img_mat, selected_coordinates, selected_labels, patch, '{0}_initial'.format(date), path)
        #sampling
        Init_coor, Init_label = Get_intrinal_sample(os.path.join(path,"{0}/WHU_{1}_sampling.txt".format(date, date)))
        selected_coordinates = np.array(Init_coor)
        selected_labels = np.array(Init_label)
        Classification(img_mat, selected_coordinates, selected_labels, patch, '{0}_sampling'.format(date), path)
