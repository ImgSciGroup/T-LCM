from osgeo import gdal
from osgeo import osr
import numpy as np

'''
.npy to .txt
'''

def getSRSPair(dataset):
    '''
    获得给定数据的投影参考系和地理参考系
    :param dataset: GDAL地理数据
    :return: 投影参考系和地理参考系
    '''
    prosrs = osr.SpatialReference()
    PROJ = dataset.GetProjection()
    prosrs.ImportFromWkt(PROJ)
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs


def geo2lonlat(dataset, x, y):
    '''
    将投影坐标转为经纬度坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param x: 投影坐标x
    :param y: 投影坐标y
    :return: 投影坐标(x, y)对应的经纬度坐标(lon, lat)
    '''

    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(prosrs, geosrs)
    coords = ct.TransformPoint(x, y)
    return coords[:2]


def lonlat2geo(dataset, lon, lat):
    '''
    将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param lon: 地理坐标lon经度
    :param lat: 地理坐标lat纬度
    :return: 经纬度坐标(lon, lat)对应的投影坐标
    '''

    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(geosrs, prosrs)
    coords = ct.TransformPoint(lon, lat)
    return coords[:2]


def imagexy2geo(dataset, row, col):
    '''
    根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
    :param dataset: GDAL地理数据
    :param row: 像素的行号
    :param col: 像素的列号
    :return: 行列号(row, col)对应的投影坐标或地理坐标(x, y)
    '''

    trans = dataset.GetGeoTransform()
    px = trans[0] + col * trans[1] + row * trans[2]
    py = trans[3] + col * trans[4] + row * trans[5]
    return px, py


def geo2imagexy(dataset, x, y):
    '''
    根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param x: 投影或地理坐标x
    :param y: 投影或地理坐标y
    :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    '''

    trans = dataset.GetGeoTransform()
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解

def write_txt(coord, label, date):
    with open('../Data/RSE/WHU_{}.txt'.format(date), 'w') as train:
        for i in range(len(coord)):
                train.write('{0}'.format(label[i]+1) + "," + '{0}'.format(coord[i][1]) + ","+'{0}'.format(coord[i][0]) + '\n')
    train.close()

if __name__ == '__main__':
    #Wuhan#
    dates = [201812, 201901,201902,201903,201904,201905,201906,201907,201908,201909,201910,201911,201912,
             202001,202002,202003,202004,202005,202006,202007,202008,202009,202010,202011,202012,
             202101, 202102, 202103, 202104, 202105, 202106, 202107, 202108, 202109, 202110, 202111, 202112,
             202201,202202,202203,202204,202205,202206,202207,202208,202209,202210,202211]
    data = np.load("D:\Paper08-20240515-RSE\Data\RSE\WUHAN\Wuhan.npy")
    for date in range(len(dates)):
        result = data[:, -3:, date:date+1]
        result = np.squeeze(result, axis=2)
        label = result[:, 0]
        coordinate = result[:, 1:]
        gdal.AllRegister()
        dataset = gdal.Open(r"D:\Paper08-20240515-RSE\Data\WuHan\\{}.tif".format(dates[date]))

        print('数据投影：')
        print(dataset.GetProjection())
        print('数据的大小（行，列）：')
        print('(%s %s)' % (dataset.RasterYSize, dataset.RasterXSize))

        coord = []
        for i in range(len(label)):

            # print('经纬度 -> 投影坐标：')
            coords = lonlat2geo(dataset, coordinate[i][0], coordinate[i][1])

            # print('投影坐标 -> 图上坐标：')
            coords = geo2imagexy(dataset, coords[0], coords[1])

            coord.append(coords)

        print(1)
        Init_coor = np.array(coord)
        write_txt(np.round(coord).astype(int), np.round(label).astype(int), date)
        print(1)