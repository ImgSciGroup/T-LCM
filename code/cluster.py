
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import stats
import os
def Cluster_spectral(coordinate_array, img, id):
    Data = []
    for i in range(len(coordinate_array)):
        coord = coordinate_array[i]
        Data.append(img[coord[0], coord[1]])
    Data = np.array(Data)

    #1.根据概率密度绘制样本分布
    kde = stats.gaussian_kde([Data[:,0],Data[:,1]])
    x_min, x_max = Data[:, 0].min(), Data[:, 0].max()
    y_min, y_max = Data[:, 1].min(), Data[:, 1].max()
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])
    z = kde(grid_coords).reshape(x_grid.shape)

    # 绘制等高线图
    plt.figure()

    # contour = plt.contour(x_grid, y_grid, z, levels=10, cmap='viridis')
    plt.contourf(x_grid, y_grid, z, levels=10, cmap='viridis')
    plt.scatter(Data[:, 0], Data[:, 1], s=2, label='Samples')
    plt.xlabel('Dimension-1')
    plt.ylabel('Dimension-2')
    plt.title('Spectral Density Estimation and Contour Plot')
    plt.legend()
    plt.colorbar(label='Density')
    plt.savefig("D:\Paper08-20240515-RSE\Data\PC\Class-{0}\Spectral Density Estimation and Contour Plot\\Figure-1.png".format(id), dpi=500)
    # plt.show()


def Cluster_spatial(coordinate_array, img, id):
    #1.根据概率密度绘制样本分布
    kde = stats.gaussian_kde([coordinate_array[:,0],coordinate_array[:,1]])
    x_min, x_max = coordinate_array[:, 0].min(), coordinate_array[:, 0].max()
    y_min, y_max = coordinate_array[:, 1].min(), coordinate_array[:, 1].max()
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])
    z = kde(grid_coords).reshape(x_grid.shape)

    # 绘制等高线图
    plt.figure()
    # contour = plt.contour(x_grid, y_grid, z, levels=10, cmap='viridis')
    contour = plt.contourf(x_grid, y_grid, z, levels=10, cmap='viridis')
    plt.scatter(coordinate_array[:, 0], coordinate_array[:, 1], s=1, label='Samples')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Spatial Density Estimation and Contour Plot')
    plt.legend()
    plt.colorbar(label='Density')
    plt.savefig("D:\Paper08-20240515-RSE\Data\PC\Class-{0}\Spatial Density Estimation and Contour Plot\\Figure-1.png".format(id), dpi=500)

    # plt.show()
    # max_z = max(z)
    # min_z = min(z)
    #
    # # 获取等高线级别区域的数据
    # def get_coords_in_contour(paths, data_coords):
    #     result_coords = []
    #     for path in paths:
    #         for coord in data_coords:
    #             if path.contains_point(coord):  # 检查数据点是否在等高线路径内
    #                 result_coords.append(coord)
    #     return result_coords
    #
    # # 存储等高线级别区域的数据
    # contour_data = {}
    #
    # for level_idx, level in enumerate(contour.collections):
    #     paths = level.get_paths()  # 获取等高线路径
    #     coords_in_level = get_coords_in_contour(paths, coordinate_array[:,:2])  # 获取在当前等高线级别内的数据点
    #     contour_data[contour.levels[level_idx]] = coords_in_level  # 将数据点存储在字典中
    #
    # # 创建直方图
    # density_values = kde([coordinate_array[:,0],coordinate_array[:,1]])  # 计算数据点的密度估计值
    # bins = np.histogram_bin_edges(density_values, bins='auto')  # 自动确定直方图的bin边界
    #
    # # 生成直方图
    # plt.figure(figsize=(10, 8))  # 创建新的图形
    # plt.hist(density_values, bins=bins, color='blue', edgecolor='black', alpha=0.7)  # 绘制直方图
    # plt.title('Histogram of KDE Values')  # 设置标题
    # plt.xlabel('Density Value')  # 设置X轴标签
    # plt.ylabel('Frequency')  # 设置Y轴标签
    # plt.show()  # 显示图形
    #
    #
    # # 输出每个等高线级别区域的数据对应坐标
    # for level, coords in contour_data.items():
    #     print(f"\nContour Level {level}:")
    #     if len(coords) > 0:
    #         for coord in coords:
    #             print(f"  Coordinate: ({coord[0]:.2f}, {coord[1]:.2f})")
    #     else:
    #         print("  No data points")



def Cluster_Gauss(coordinate_array, img, n, id, date, read_datapath):
    Data = []
    for i in range(len(coordinate_array)):
        coord = coordinate_array[i]
        Data.append(img[coord[0], coord[1]])
    Data = np.array(Data)

    #2.根据正态分布绘制样本分布
    # 进行KMeans聚类
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(Data[:,:2])
    center = kmeans.cluster_centers_

    # 计算每个点到最近聚类中心的欧式距离

    distances = np.sqrt(np.sum((Data[:,:2] - center[0]) ** 2, axis=1))
    # 计算距离的均值和标准差
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    # 分为三部分
    part1 = Data[distances <= mean_distance + std_distance]
    part2 = Data[(distances > mean_distance + std_distance) & (distances <= mean_distance + 2 * std_distance)]
    part3 = Data[distances > mean_distance + 2 * std_distance]

    # 计算每个区域的样本个数
    part1_count = part1.shape[0]
    part2_count = part2.shape[0]
    part3_count = part3.shape[0]

    # 定义颜色
    color1 = 'blue'
    color2 = 'green'
    color3 = 'red'

    # 创建背景网格
    x_min, x_max = np.min(Data[:, 0]) - 0.1, np.max(Data[:, 0]) + 0.1
    y_min, y_max = np.min(Data[:, 1]) - 0.1, np.max(Data[:, 1]) + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    # 渲染背景颜色
    bg_color = np.zeros((xx.shape[0], xx.shape[1], 3))
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            point = np.array([xx[i, j], yy[i, j]])
            distance = np.linalg.norm(point - center)
            if distance <= mean_distance + std_distance:
                bg_color[i, j] = [0.68, 0.85, 0.90]  # lightblue
            elif distance <= mean_distance + 2 * std_distance:
                bg_color[i, j] = [0.56, 0.93, 0.56]  # lightgreen
            else:
                bg_color[i, j] = [0.94, 0.50, 0.50]  # lightcoral

    # 绘制背景颜色
    plt.figure()
    plt.imshow(bg_color, extent=(x_min, x_max, y_min, y_max), origin='lower', alpha=0.3)

    # 绘制散点图
    plt.scatter(part1[:, 0], part1[:, 1], c=color1, label='[μ−σ,μ+σ]')
    plt.scatter(part2[:, 0], part2[:, 1], c=color2, label='[μ−2σ,μ−σ]∪[μ+σ,μ+2σ]')
    plt.scatter(part3[:, 0], part3[:, 1], c=color3, label='[−∞,μ−2σ]∪[μ+2σ,+∞]')


    # 绘制中心点
    plt.scatter(center[0][0], center[0][1], color='black', marker='x', label='Center Point')

    # 添加图例和标签
    plt.legend()
    plt.xlabel('Dimension-1')
    plt.ylabel('Dimension-2')
    plt.legend()
    save_folder = os.path.join(read_datapath,"Class-{0}/Kmean Cluster Estimation and Contour Plot".format(id))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(os.path.join(save_folder, 'Figure-1.png'),
                dpi=500)
    # plt.show()

    # 绘制直方图
    counts = [part1_count, part2_count, part3_count]
    labels = ['[μ−σ,μ+σ]', '[μ−2σ,μ−σ]∪[μ+σ,μ+2σ]', '[−∞,μ−2σ]∪[μ+2σ,+∞]']
    colors = [color1, color2, color3]
    plt.figure()
    plt.bar(labels, counts, color=colors)
    plt.title('Sample Counts in Each Region')
    plt.xlabel('Region')
    plt.ylabel('Sample Count')
    save_folder = os.path.join(read_datapath,"Class-{0}/Kmean Cluster Estimation and Contour Plot".format(id))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(os.path.join(save_folder, 'Figure-2.png'),
                dpi=500)
    # plt.show()

    # 计算每个区域的样本选择概率
    # total_count = part1_count + part2_count + part3_count
    # prob_part1 = part1_count / total_count
    # prob_part2 = part2_count / total_count
    # prob_part3 = part3_count / total_count

    # 按反比例随机选取100个样本
    selected_samples = []
    for _ in range(n):
        region_choice = np.random.choice(['part1', 'part2', 'part3'], p=[0.7, 0.2, 0.1])
        if region_choice == 'part1':
            idx = np.random.randint(part1.shape[0])
            selected_samples.append(coordinate_array[distances <= mean_distance + std_distance][idx])
        elif region_choice == 'part2':
            idx = np.random.randint(part2.shape[0])
            selected_samples.append(
                coordinate_array[(distances > mean_distance + std_distance) & (distances <= mean_distance + 2 * std_distance)][
                    idx])
        else:
            if part3.shape[0]<=0:
                continue
            else:
                idx = np.random.randint(part3.shape[0])
                selected_samples.append(coordinate_array[distances > mean_distance + 2 * std_distance][idx])

    # 返回这100个样本的二维坐标
    selected_samples = np.array(selected_samples)
    print("Selected samples coordinates:\n", selected_samples)
    label = np.full(selected_samples.shape[0], id)
    return selected_samples, label

