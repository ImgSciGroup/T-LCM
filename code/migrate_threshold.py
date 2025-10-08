
import linecache
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw
import os
def Get_intrinal_sample(Path):
    Init_label = []
    Init_coor = []
    for j in range(1, len(open(Path).readlines()) + 1):
        lindata = linecache.getline(Path, j)
        data_label,data_x, data_y = int(lindata.split(',')[0]), int(lindata.split(',')[1]), int(lindata.split(',')[2])
        Init_coor.append([data_x, data_y])
        Init_label.append(data_label)
    return Init_coor,Init_label

def get_class_data(image_data, coords):
    class_data = []
    for img_idx, (x, y) in enumerate(coords):
        class_data.append(image_data[x, y,:])
    return np.array(class_data)
def filter_samples(thresholds, coords, labels, image1, image2):
    filtered_coords = []
    filtered_labels = []

    for idx, (coord, label) in enumerate(zip(coords, labels)):
        x, y = coord
        vector1 = image1[x, y, :].astype(np.float64)
        vector2 = image2[x, y, :].astype(np.float64)

        distance = dtw.distance_fast(vector1.astype(np.float64), vector2.astype(np.float64))

        if distance < thresholds[label - 1]:
            filtered_coords.append(coord)
            filtered_labels.append(label)

    return np.array(filtered_coords), np.array(filtered_labels)


def get_top_n_by_class(coords, labels, number_migrate, n):
    top_coords = []
    top_labels = []
    unique_labels = np.unique(labels)

    for label in unique_labels:
        indices = np.where(labels == label)[0]

        if len(indices) <= (n[label-1]/np.sum(n))*number_migrate:
            selected_indices = indices
        else:
            selected_indices = np.random.choice(indices, size=(int((n[label-1]/np.sum(n))*number_migrate)), replace=False)

        top_coords.extend(coords[selected_indices])
        top_labels.extend(labels[selected_indices])

    return np.array(top_coords), np.array(top_labels)

def write_txt(coord, label, id, path):
    with open(os.path.join(path,'WHU_{0}.txt'.format(id)), 'w') as train:
        for i in range(len(coord)):
                train.write('{0}'.format(label[i]) + "," + '{0}'.format(coord[i][0]) + ","+'{0}'.format(coord[i][1]) + '\n')
    train.close()

def migrate_threshold(date1, date2, image1, image2, number_migrate, ratio, verif_coor, verif_label, Date1_Intermediate_result, Date2_Intermediate_result):
    Init_coor, Init_label = Get_intrinal_sample(
        os.path.join(Date1_Intermediate_result, "WHU_{0}_initial.txt".format(date1)))
    Init_coor = np.array(Init_coor)
    Init_label = np.array(Init_label)
    coords_class1,coords_class2,coords_class3,coords_class4,coords_class5,coords_class6= [[] for x in range(6)]
    # 将坐标放入列表中
    for i in range(1, 7):
        if i==1:
            coords_class1 = Init_coor[Init_label == i]
        if i==2:
            coords_class2 = Init_coor[Init_label == i]
        if i==3:
            coords_class3 = Init_coor[Init_label == i]
        if i==4:
            coords_class4 = Init_coor[Init_label == i]
        if i==5:
            coords_class5 = Init_coor[Init_label == i]
        if i==6:
            coords_class6 = Init_coor[Init_label == i]


    coords_classes = [coords_class1, coords_class2, coords_class3, coords_class4, coords_class5, coords_class6]

    class_data = [get_class_data(image1, coords) for coords in coords_classes]

    # 计算类间的距离矩阵
    num_classes = len(class_data)

    #####DWT最小距离#########
    # min_distances = np.zeros((num_classes, num_classes))
    # for i in range(num_classes):
    #     for j in range(num_classes):
    #         if i != j:
    #             min_dist = np.inf
    #             for vector_i in class_data[i]:  # 确保数据类型为 float32
    #                 for vector_j in class_data[j]:  # 确保数据类型为 float32
    #                     distance = dtw.distance_fast(vector_i.astype(np.float64), vector_j.astype(np.float64))
    #                     if distance < min_dist:
    #                         min_dist = distance
    #             min_distances[i, j] = min_dist

    #####DWT平均距离#########
    average_distances = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                distances = []
                for vector_i in class_data[i]:
                    for vector_j in class_data[j]:
                        distance = dtw.distance_fast(vector_i.astype(np.float64), vector_j.astype(np.float64))
                        distances.append(distance)
                average_distances[i, j] = np.mean(distances)

    sorted_indices = np.argsort(average_distances, axis=1)
    threshold = average_distances[np.arange(average_distances.shape[0]), sorted_indices[:, 1]]
    # 绘制类间归一化距离
    plt.figure(figsize=(8, 8))
    plt.imshow(average_distances, interpolation='nearest', cmap='viridis')
    plt.colorbar()
    plt.title('Normalized Distances Between Class Means')
    plt.xlabel('Class')
    plt.ylabel('Class')
    plt.xticks(np.arange(6), ['Water body', 'Woodland', 'Grassland', 'Bare soil', 'Impervious', 'Cropland'])
    plt.yticks(np.arange(6), ['Water body', 'Woodland', 'Grassland', 'Bare soil', 'Impervious', 'Cropland'])
    # 标记每个方格中的距离值
    for i in range(average_distances.shape[0]):
        for j in range(average_distances.shape[1]):
            plt.text(j, i, f'{average_distances[i, j]:.2f}', ha='center', va='center',
                     color='white' if average_distances[i, j] < 0.5 else 'black')

    plt.tight_layout()
    plt.savefig(os.path.join(Date1_Intermediate_result,"Figure-{}.png".format(date1)),
                dpi=500)
    # plt.show()


    filtered_coords, filtered_labels = filter_samples(threshold, verif_coor, verif_label, image1, image2)
    top_coords, top_labels = get_top_n_by_class(filtered_coords, filtered_labels, number_migrate, ratio)
    write_txt(top_coords, top_labels, '{0}_initial'.format(date2), Date2_Intermediate_result)