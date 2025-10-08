import numpy as np
from dtaidistance import dtw
from scipy.spatial.distance import euclidean
import tifffile
import linecache
def get_vectors(image1, image2, coords):
    vectors1 = np.array([image1[x, y, :] for x, y in coords])
    vectors2 = np.array([image2[x, y, :] for x, y in coords])
    return vectors1, vectors2

def Get_intrinal_sample(Path):
    Init_label = []
    Init_coor = []
    for j in range(1, len(open(Path).readlines()) + 1):
        lindata = linecache.getline(Path, j)
        data_label,data_x, data_y = int(lindata.split(',')[0]), int(lindata.split(',')[1]), int(lindata.split(',')[2])
        Init_coor.append([data_x, data_y])
        Init_label.append(data_label)
    return Init_coor,Init_label
def compute_dtw_distances(vectors1, vectors2):
    distances = np.array([dtw.distance_fast(v1, v2) for v1, v2 in zip(vectors1.astype(np.float64), vectors2.astype(np.float64))])
    return distances
def write_txt(coord, label, id):
    with open('../Data/RSE/WHU_{0}.txt'.format(id), 'w') as train:
        for i in range(len(coord)):
                train.write('{0}'.format(label[i]) + "," + '{0}'.format(coord[i][0]) + ","+'{0}'.format(coord[i][1]) + '\n')
    train.close()


def get_top_n_by_class(distances, coords, labels, n, number_migrate):
    top_coords = []
    top_labels = []
    unique_labels = np.unique(labels)

    selected_indices = set()

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        label_distances = distances[indices]
        label_coords = coords[indices]

        sorted_indices = np.argsort(label_distances)

        # 保证选取不重复的坐标和标签
        count = 0
        for idx in sorted_indices:
            if count >= (n[label-1]/np.sum(n))*number_migrate:
                break
            if indices[idx] not in selected_indices:
                top_coords.append(label_coords[idx])
                top_labels.append(labels[indices][idx])
                selected_indices.add(indices[idx])
                count += 1

    return np.array(top_coords), np.array(top_labels)


# 示例输入
date1 = '201812(1)'
date2 = '201901(1)'
image1 = tifffile.imread("D:\Paper08-20240515-RSE\Data\RSE\WUHAN\\{0}.tif".format(date1))
image2 = tifffile.imread("D:\Paper08-20240515-RSE\Data\RSE\WUHAN\\{0}.tif".format(date2))
Init_coor, Init_label = Get_intrinal_sample("../Data/RSE/WHU_201812_sampling.txt")
coords = np.array(Init_coor)
labels = np.array(Init_label)
number_migrate = 2000



# 提取对应位置的1x10向量
vectors1, vectors2 = get_vectors(image1, image2, coords)

# 计算每个坐标点在两个时间点之间的距离
distances = compute_dtw_distances(vectors1, vectors2)

# 获取每类距离最小的50个标签和坐标
n= ratio = [1618, 465, 26, 49, 1885, 2079]
#WuHan[1618, 465, 26, 49, 1885, 2079]#
top_coords, top_labels = get_top_n_by_class(distances, coords, labels, n, number_migrate)
write_txt(top_coords, top_labels, '{0}_initial'.format(date2))
