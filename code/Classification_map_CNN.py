
import keras
from keras.layers import Flatten, Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from matplotlib import pyplot as plt
import linecache
import tifffile
import os
import numpy as np
import cv2
from keras.utils import to_categorical



########################################################CNN classification##################################################
def Get_intrinal_sample(Path):
    Init_label = []
    Init_coor = []
    for j in range(1, len(open(Path).readlines()) + 1):
        lindata = linecache.getline(Path, j)
        data_label,data_x, data_y = int(lindata.split(',')[0]), int(lindata.split(',')[1]), int(lindata.split(',')[2])
        Init_coor.append([data_x, data_y])
        Init_label.append(data_label)
    return Init_coor,Init_label


# 提取5x5x10 patch并将其平均为1x10向量
def extract_patch_and_average(padded_image, coord, patch_size):
    half_size = patch_size // 2
    x, y = coord
    x += half_size  # 调整坐标以匹配填充后的图像
    y += half_size
    patch = padded_image[x - half_size:x + half_size + 1, y - half_size:y + half_size + 1]
    return patch





if __name__ == '__main__':
    Time = ['201812', '201901', '201902', '201903', '201904', '201905', '201906', '201907', '201908','201909', '201910','201911', '201912',
            '202001', '202002', '202003', '202004', '202005', '202006', '202007', '202008', '202009', '202010','202011', '202012',
            '202101', '202102', '202103', '202104', '202105', '202106', '202107', '202108', '202109', '202110','202111', '202112',
            '202201', '202202', '202203', '202204', '202205', '202206', '202207', '202208', '202209', '202210','202211']
    img_path = '../Data/WuHan'
    path = '../Data/WuHan\Intermediate results\SVM\\120\\4000'
    classes=7
    patch = 7
    shape = (7, 7, 10)


    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=shape))
    model.add(BatchNormalization())
    model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(classes, activation='softmax'))
    model.compile(loss=categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])


    for date in Time:
        image = tifffile.imread(os.path.join(img_path,"{}.tif".format(date)))
        #initial
        Init_coor, Init_label = Get_intrinal_sample(os.path.join(path,"{0}/WHU_{1}_initial.txt".format(date, date)))
        coords = np.array(Init_coor)
        labels = np.array(Init_label)
        labels = to_categorical(labels)
        # 添加填充以确保提取patch时不会超出图像范围
        padding = int(patch / 2)  # 对应于patch的一半尺寸（5x5的patch则padding为2）
        padded_image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')

        # 提取所有坐标的patch并平均
        patches = np.array([extract_patch_and_average(padded_image, coord, patch) for coord in coords])

        histoty = model.fit(patches, labels, epochs=200, batch_size=64)

        # plt.plot(histoty.history['loss'], label='loss')
        # plt.legend()
        # plt.show()
        # plt.plot(histoty.history['accuracy'], label='acc')
        # plt.legend()
        # plt.show()
        classification_map = np.zeros((image.shape[0], image.shape[1]), dtype=int)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                patch_vector = padded_image[i:i + patch, j:j + patch, :]
                patch_vector = patch_vector.reshape([1, patch_vector.shape[0], patch_vector.shape[1], patch_vector.shape[2]])
                classification_map[i, j] = np.argmax(model.predict(patch_vector), axis=1)
        cv2.imwrite(os.path.join(img_path, "Result/{0}_{1}_result.bmp".format(date, 'Cnn')), classification_map)

