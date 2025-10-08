import cv2
import numpy as np
import tifffile
import linecache

def Get_intrinal_sample(Path):
    Init_label = []
    Init_coor = []
    for j in range(1, len(open(Path).readlines()) + 1):
        lindata = linecache.getline(Path, j)
        data_label,data_x, data_y = int(lindata.split(',')[0]), int(lindata.split(',')[1]), int(lindata.split(',')[2])
        Init_coor.append([data_x, data_y])
        Init_label.append(data_label)
    return Init_coor,Init_label

def write_txt(coord, label, id):
    with open('../Data/WuHan/WHU_{0}.txt'.format(id), 'w') as train:
        for i in range(len(coord)):
                train.write('{0}'.format(label[i]) + "," + '{0}'.format(coord[i][0]) + ","+'{0}'.format(coord[i][1]) + '\n')
    train.close()

if __name__ == '__main__':
    #########读数据和参数设置###########
    date = '202211'
    img = cv2.imread("../Data/WuHan/{}-RGB.tif".format(date))
    img_mat = tifffile.imread("../Data/WuHan/{}.tif".format(date))
    num_class = 6
    initial_num_sample = 20

    Init_coor, Init_label = Get_intrinal_sample("../Data/WuHan/verfication/WHU_{}.txt".format(date))
    Init_coor = np.array(Init_coor)
    Init_label = np.array(Init_label)


    # 创建存储每类中随机选取的20个坐标及其对应标签的列表
    selected_coordinates = []
    selected_labels = []
    for class_label in range(1, num_class+1):
        class_indices = np.where(Init_label == class_label)[0]
        print(len(class_indices))
        if len(class_indices) >= initial_num_sample:
            selected_indices = np.random.choice(class_indices, initial_num_sample, replace=False)
        else:
            selected_indices = class_indices  # 如果少于20个样本，选择所有样本
        selected_coordinates.append(Init_coor[selected_indices])
        selected_labels.append(Init_label[selected_indices])


    # 将随机选取的20个坐标及其对应标签转换为numpy数组
    selected_coordinates = np.vstack(selected_coordinates)
    selected_labels = np.hstack(selected_labels)
    # 选取的20个样本存入txt
    write_txt(selected_coordinates, selected_labels, '{0}_initial_{1}'.format(date,initial_num_sample))