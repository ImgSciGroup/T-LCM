import os.path

import numpy as np
import cv2
import linecache
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import tifffile
from cluster import Cluster_Gauss as Cluster
from migrate_threshold import migrate_threshold
def Get_intrinal_sample(Path):
    Init_label = []
    Init_coor = []
    for j in range(1, len(open(Path).readlines()) + 1):
        lindata = linecache.getline(Path, j)
        data_label,data_x, data_y = int(lindata.split(',')[0]), int(lindata.split(',')[1]), int(lindata.split(',')[2])
        Init_coor.append([data_x, data_y])
        Init_label.append(data_label)
    return Init_coor,Init_label

def extract_patch(image, coord, patchsize):
    """
    提取以给定像素坐标为中心的3x3 patch。

    参数:
    image -- 输入图像，形状为 (H, W) 或 (H, W, C)
    coord -- 像素坐标，形状为 (N, 2)

    返回:
    patches -- 形状为 (N, 3, 3) 或 (N, 3, 3, C) 的patch数组
    """
    patches = []
    H, W = image.shape[:2]

    for (x, y) in coord:
        # 计算patch的起始和结束位置
        x_start, x_end = max(x - int(patchsize/2), 0), min(x + int(patchsize/2)+1, H)
        y_start, y_end = max(y - int(patchsize/2), 0), min(y + int(patchsize/2)+1, W)

        # 提取patch
        patch = image[x_start:x_end, y_start:y_end]

        # 如果patch不是3x3的大小，则填充0
        if patch.shape[0] < patchsize or patch.shape[1] < patchsize:
            patch_padded = np.zeros((patchsize, patchsize) + patch.shape[2:], dtype=image.dtype)
            patch_padded[:patch.shape[0], :patch.shape[1]] = patch
            patch = patch_padded
        patch_mean = patch.mean(axis=(0, 1))
        patches.append(patch_mean)

    return np.array(patches)

def first_difference(img, T, interval):
    visit_map = np.zeros(img.shape[:2])
    gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    for i in range(0, gray_img.shape[0], interval):
        row_array = gray_img[i,:]
        for j in range(1, len(row_array)):
            if int(row_array[j]-row_array[j-1])<T:
                visit_map[i,j-1:j+1]=1

    for i in range(0, gray_img.shape[1], interval):
        col_array = gray_img[:,i]
        for j in range(1, len(col_array)):
            if int(col_array[j]-col_array[j-1])<T:
                visit_map[j-1:j+1, i] =1

    # cv2.imwrite("./visit_map.tif", visit_map)
    return visit_map

def get_patch_mean(image, x, y, patchsize):
    """
    获取以 (x, y) 为中心的 3x3 patch 的均值

    参数:
    image -- 形状为 (H, W, D) 的图像
    x -- 中心像素的 x 坐标
    y -- 中心像素的 y 坐标

    返回:
    patch_mean -- patch 的均值，形状为 (D,)
    """
    rows, cols, _ = image.shape

    # 确定 patch 的边界
    x_start = max(x - int(patchsize/2), 0)
    x_end = min(x + int(patchsize/2)+1, rows)
    y_start = max(y - int(patchsize/2), 0)
    y_end = min(y + int(patchsize/2)+1, cols)

    # 获取 3x3 patch
    patch = image[x_start:x_end, y_start:y_end, :]
    if patch.shape[0] < patchsize or patch.shape[1] < patchsize:
        patch_padded = np.zeros((patchsize, patchsize) + patch.shape[2:], dtype=image.dtype)
        patch_padded[:patch.shape[0], :patch.shape[1]] = patch
        patch = patch_padded
    patch = patch.mean(axis=(0, 1))
    return patch

def get_accessible_coords_and_class(pixel_coords, visitmap, spectral_image, train_patch, Init_label, patchsize, a):
    """
    获取在 visitmap 上落在方向线上的所有坐标，并计算其 3x3 patch 的类别概率。

    参数:
    pixel_coords -- 像素坐标数组，形状为 (N, 2)
    visitmap -- 访问图，形状为 (H, W)，1 表示可访问，0 表示禁止访问
    spectral_image -- 光谱图像，形状为 (H, W, D)
    class_vectors -- 类别向量，形状为 (C, D)

    返回:
    result -- 包含坐标和其最大概率类号的列表 [(x, y, class_idx), ...]
    """
    # train_patch = train_patch.reshape(train_patch.shape[0], -1)
    svm_model = SVC(kernel='rbf', C=10.0, probability=True)
    svm_model.fit(train_patch, Init_label)

    rows, cols, d = spectral_image.shape

    if a==45:
    # 定义八个方向
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    if a==90:
    # 定义八个方向
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]




    Augment_coordinates = []
    Augment_labels = []


    for x, y in pixel_coords:
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            while 0 <= nx < rows and 0 <= ny < cols:
                if visitmap[nx, ny] == 1:
                    # 获取3x3 patch的光谱均值
                    patch = get_patch_mean(spectral_image, nx, ny, patchsize)
                    patch = patch.reshape(1, -1)
                    # svm预测
                    probabilities = svm_model.predict_proba(patch)

                    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
                    if entropy<=1.4:
                        # 获取最大概率的类号
                        class_idx = np.argmax(probabilities)

                        Augment_coordinates.append([nx, ny])
                        Augment_labels.append(class_idx+1)

                nx += dx
                ny += dy
                if nx < 0 or nx >= rows or ny < 0 or ny >= cols:
                    break

    return np.array(Augment_coordinates), np.array(Augment_labels)
def get_accessible_coords_and_class1(pixel_coords, visitmap, spectral_image, train_patch, Init_label, patchsize, angle):
    """
    获取在 visitmap 上落在方向线上的所有坐标，并计算其 3x3 patch 的类别概率。

    参数:
    pixel_coords -- 像素坐标数组，形状为 (N, 2)
    visitmap -- 访问图，形状为 (H, W)，1 表示可访问，0 表示禁止访问
    spectral_image -- 光谱图像，形状为 (H, W, D)
    class_vectors -- 类别向量，形状为 (C, D)

    返回:
    result -- 包含坐标和其最大概率类号的列表 [(x, y, class_idx), ...]
    """

    # 定义方向
    max_length = 999999
    angles = np.arange(0, 360, angle)
    svm_model = SVC(kernel='rbf', C=10.0, probability=True)
    svm_model.fit(train_patch, Init_label)
    rows, cols, d = spectral_image.shape

    Augment_coordinates = []
    Augment_labels = []

    for x, y in pixel_coords:
        for angle_l in angles:
            for length in range(1, max_length + 1):
                new_x = x + int(length * np.cos(np.radians(angle_l)))
                new_y = y + int(length * np.sin(np.radians(angle_l)))
                if 0 <= new_x < rows and 0 <= new_y < cols:
                    if visitmap[new_x, new_y] == 1:
                        # 获取3x3 patch的光谱均值
                        patch = get_patch_mean(spectral_image, new_x, new_y, patchsize)
                        patch = patch.reshape(1, -1)
                        # svm预测
                        probabilities = svm_model.predict_proba(patch)

                        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
                        if entropy<=1.4:
                            # 获取最大概率的类号
                            class_idx = np.argmax(probabilities)

                            Augment_coordinates.append([new_x, new_y])
                            Augment_labels.append(class_idx+1)
                else:
                    break

    return np.array(Augment_coordinates), np.array(Augment_labels)
def draw_circles(image, pixel_coords, class_labels, radius=5, thickness=5):
    """
    在图像上绘制圈，使用不同颜色根据类标签区分。

    参数:
    image -- 输入图像，形状为 (H, W, C)
    pixel_coords -- 像素坐标数组，形状为 (N, 2)
    class_labels -- 类标签数组，形状为 (N,)
    colors -- 颜色数组，形状为 (9, 3)
    radius -- 圆的半径，默认为5
    thickness -- 圆的边框厚度，默认为2

    返回:
    image_with_circles -- 带有绘制圈的图像
    """
    pixel_coords = pixel_coords.tolist()
    class_labels = class_labels.tolist()
    colors = np.array([

            [255, 184, 88],
            [31,70,25],
            [27, 208, 138],
            [128, 168, 222],
            [56, 67, 212],
            [156, 214, 255],
            [173, 222, 255]



    ])

    image_with_circles = image.copy()

    for (x, y), label in zip(pixel_coords, class_labels):
        color = colors[label-1]
        cv2.circle(image_with_circles, (y, x), radius,  (int(color[0]), int(color[1]), int(color[2])), thickness)

    return image_with_circles

def write_txt(coord, label, id, path):
    with open(os.path.join(path,'WHU_{0}.txt'.format(id)), 'w') as train:
        for i in range(len(coord)):
                train.write('{0}'.format(label[i]) + "," + '{0}'.format(coord[i][0]) + ","+'{0}'.format(coord[i][1]) + '\n')
    train.close()



if __name__ == '__main__':
    #########读数据和参数设置###########



    date1 = '201812'
    Dates2 = ['201902', '201903', '201904', '201905', '201906', '201907', '201908', '201909', '201910','201911', '201912',
            '202001', '202002', '202003', '202004', '202005', '202006', '202007', '202008', '202009', '202010', '202011', '202012',
            '202101', '202102', '202103', '202104', '202105', '202106', '202107', '202108', '202109', '202110', '202111', '202112',
            '202201', '202202', '202203', '202204', '202205', '202206', '202207', '202208', '202209', '202210', '202211']

    read_datapath = '../Data/WuHan'
    Classification_id = 'SVM'
    patch = 7
    num_class = 6
    num_sampling_samples = [4000]
    a_s = [30]
    number_migrate = 2000
    ratio = [1500, 500, 200, 300, 1500, 2000]
    Initial_sample = [40, 60, 80, 100]
    ratio1 = ratio/np.sum(ratio)
    #WuHan[1618, 465, 26, 49, 1885, 2079]#
    for num_sampling_sample in num_sampling_samples:
        for a in a_s:
            for initial_sample in Initial_sample:
                for date2 in Dates2:
                    img1_rgb = cv2.imread(os.path.join(read_datapath, "RGB/{}-RGB.tif".format(date1)))
                    img1_tif = tifffile.imread(os.path.join(read_datapath, "{}.tif".format(date1)))

                    img2_rgb = cv2.imread(os.path.join(read_datapath, "RGB/{}-RGB.tif".format(date2)))
                    img2_tif = tifffile.imread(os.path.join(read_datapath, "{}.tif".format(date2)))

                    #建立第一时相中间结果文件夹#
                    Date1_Intermediate_result = os.path.join(read_datapath, 'Intermediate results', Classification_id,'{0}'.format(a), '{0}'.format(num_sampling_sample), 'Num_sample','{0}'.format(initial_sample),'{0}'.format(date1))
                    if not os.path.exists(Date1_Intermediate_result):
                        os.makedirs(Date1_Intermediate_result)

                    #读第一时相初始样本txt#
                    Init_coor, Init_label = Get_intrinal_sample(os.path.join(Date1_Intermediate_result,"WHU_{0}_initial.txt".format(date1)))
                    selected_coordinates = np.array(Init_coor)
                    selected_labels = np.array(Init_label)

                    #选取的样本标记在第一时相图上#
                    Initial_draw_circles = draw_circles(img1_rgb, selected_coordinates, selected_labels)
                    cv2.imwrite(os.path.join(Date1_Intermediate_result,"{0}_initial_draw.tif".format(date1)), Initial_draw_circles)

                    # 选取的初始样本进行初始样本分类
                    # Classification(img1_tif, selected_coordinates, selected_labels, patch, '{0}_initial'.format(date1))

                    ###########################PCA获取mat#############################
                    scaler = StandardScaler()
                    voxel = scaler.fit_transform(img1_tif.reshape(-1, img1_tif.shape[-1]))
                    pca = PCA(n_components=10)
                    principalComponents = pca.fit_transform(voxel)
                    img_mat = principalComponents.reshape(img1_tif.shape[0], img1_tif.shape[1], 10)
                    ###########################获取类原型、visit_map、#############################
                    train_patch = extract_patch(img_mat, selected_coordinates, patch)
                    visit_map = first_difference(img1_rgb, 2, 20)
                    if a==90 or a==45:
                        Augment_coordinates, Augment_labels = get_accessible_coords_and_class(selected_coordinates, visit_map, img_mat, train_patch, selected_labels, patch, a)
                    if a == 30 or a == 120:
                        Augment_coordinates, Augment_labels = get_accessible_coords_and_class1(selected_coordinates, visit_map, img_mat, train_patch, selected_labels, patch, a)
                    After_coor = np.concatenate((selected_coordinates, Augment_coordinates), axis=0)
                    After_label = np.concatenate((selected_labels, Augment_labels), axis=0)
                    print(1)
                    ###########################将增强未采样的样本点标记在图上#############################
                    After_draw_circles = draw_circles(img1_rgb, After_coor, After_label)
                    cv2.imwrite(os.path.join(Date1_Intermediate_result,"{0}_augmentation_draw.tif".format(date1)), After_draw_circles)
                    write_txt(After_coor, After_label, '{0}_augmentation'.format(date1), Date1_Intermediate_result)
                    ###########################聚类采样#############################
                    Second_coor = selected_coordinates
                    Second_label = selected_labels
                    for i in range(1,num_class+1):
                        coord_i = After_coor[After_label==i]
                        if int(num_sampling_sample*ratio1[i-1])<50 or len(coord_i)<50:
                            Second_coordinates = coord_i
                            Second_labels = np.full(len(coord_i), i)
                            Second_coor = np.concatenate((Second_coor, Second_coordinates), axis=0)
                            Second_label = np.concatenate((Second_label, Second_labels), axis=0)
                        else:
                            Second_coordinates, Second_labels = Cluster(coord_i, img1_rgb, int(num_sampling_sample*ratio1[i-1]),i, date1, Date1_Intermediate_result)
                            Second_coor = np.concatenate((Second_coor, Second_coordinates), axis=0)
                            Second_label = np.concatenate((Second_label, Second_labels), axis=0)

                    print(1)
                    ###########################采样后样本标记在图上#############################
                    write_txt(Second_coor, Second_label, '{0}_sampling'.format(date1), Date1_Intermediate_result)
                    Second_draw_circles = draw_circles(img1_rgb, Second_coor, Second_label)
                    cv2.imwrite(os.path.join(Date1_Intermediate_result,"{0}_sampling_draw.tif".format(date1)), Second_draw_circles)

                    # 增强采样后的样本进行分类
                    # Classification(img_mat, Second_coor, Second_label, patch, '{0}_sampling'.format(date1))

                    # 建立第二时相中间结果文件夹#
                    Date2_Intermediate_result = os.path.join(read_datapath, 'Intermediate results', Classification_id,'{0}'.format(a), '{0}'.format(num_sampling_sample),'Num_sample','{0}'.format(initial_sample),'{0}'.format(date2))
                    if not os.path.exists(Date2_Intermediate_result):
                        os.makedirs(Date2_Intermediate_result)
                    # 迁移第二时相
                    migrate_threshold(date1, date2, img1_tif, img2_tif, number_migrate, ratio, Second_coor, Second_label, Date1_Intermediate_result, Date2_Intermediate_result)


                    #读第二时相初始样本txt#
                    Init_coor, Init_label = Get_intrinal_sample(os.path.join(Date2_Intermediate_result,"WHU_{0}_initial.txt".format(date2)))
                    selected_coordinates = np.array(Init_coor)
                    selected_labels = np.array(Init_label)

                    #选取的样本标记在第二时相图上#
                    Initial_draw_circles = draw_circles(img2_rgb, selected_coordinates, selected_labels)
                    cv2.imwrite(os.path.join(Date2_Intermediate_result,"{0}_initial_draw.tif".format(date2)), Initial_draw_circles)

                    # 选取的初始样本进行初始样本分类
                    # Classification(img1_tif, selected_coordinates, selected_labels, patch, '{0}_initial'.format(date1))

                    ###########################PCA获取mat#############################
                    scaler = StandardScaler()
                    voxel = scaler.fit_transform(img1_tif.reshape(-1, img2_tif.shape[-1]))
                    pca = PCA(n_components=10)
                    principalComponents = pca.fit_transform(voxel)
                    img_mat = principalComponents.reshape(img2_tif.shape[0], img1_tif.shape[1], 10)
                    ###########################获取类原型、visit_map、#############################
                    train_patch = extract_patch(img_mat, selected_coordinates, patch)
                    visit_map = first_difference(img2_rgb, 2, 20)
                    if a == 90 or a == 45:
                        Augment_coordinates, Augment_labels = get_accessible_coords_and_class(selected_coordinates,visit_map, img_mat,train_patch, selected_labels,patch, a)
                    if a == 30 or a == 120:
                        Augment_coordinates, Augment_labels = get_accessible_coords_and_class1(selected_coordinates,visit_map, img_mat,train_patch, selected_labels,patch, a)
                    After_coor = np.concatenate((selected_coordinates, Augment_coordinates), axis=0)
                    After_label = np.concatenate((selected_labels, Augment_labels), axis=0)
                    print(1)
                    ###########################将增强未采样的样本点标记在图上#############################
                    After_draw_circles = draw_circles(img2_rgb, After_coor, After_label)
                    cv2.imwrite(os.path.join(Date2_Intermediate_result,"{0}_augmentation_draw.tif".format(date2)), After_draw_circles)
                    write_txt(After_coor, After_label, '{0}_augmentation'.format(date2), Date2_Intermediate_result)
                    ###########################聚类采样#############################
                    Second_coor = selected_coordinates
                    Second_label = selected_labels
                    for i in range(1,num_class+1):
                        coord_i = After_coor[After_label==i]
                        if int(num_sampling_sample*ratio1[i-1])<50 or len(coord_i)<50:
                            Second_coordinates = coord_i
                            Second_labels = np.full(len(coord_i), i)
                            Second_coor = np.concatenate((Second_coor, Second_coordinates), axis=0)
                            Second_label = np.concatenate((Second_label, Second_labels), axis=0)
                        else:
                            Second_coordinates, Second_labels = Cluster(coord_i, img1_rgb, int(num_sampling_sample*ratio1[i-1]),i, date2, Date2_Intermediate_result)
                            Second_coor = np.concatenate((Second_coor, Second_coordinates), axis=0)
                            Second_label = np.concatenate((Second_label, Second_labels), axis=0)

                    print(1)
                    ###########################采样后样本标记在图上#############################
                    write_txt(Second_coor, Second_label, '{0}_sampling'.format(date2), Date2_Intermediate_result)
                    Second_draw_circles = draw_circles(img2_rgb, Second_coor, Second_label)
                    cv2.imwrite(os.path.join(Date2_Intermediate_result,"{0}_sampling_draw.tif".format(date2)), Second_draw_circles)
                    print(1)