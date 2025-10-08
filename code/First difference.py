import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
def first_difference(img, T, interval):

    gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    coordinate_array = []
    for i in range(0, gray_img.shape[0], interval):
        row_array = gray_img[i,:]
        for j in range(1, len(row_array)):
            if int(row_array[j]-row_array[j-1])<T:
                img[i,j-1:j+1]=[0,0,255]
                coordinate_array.append([i,j])


    for i in range(0, gray_img.shape[1], interval):
        col_array = gray_img[:,i]
        for j in range(1, len(col_array)):
            if int(col_array[j]-col_array[j-1])<T:
                img[j-1:j+1, i] =[0,0,255]
                coordinate_array.append([j, i])
    cv2.imwrite("./visit_map.tif", img)
    return coordinate_array

def Cluster(coordinate_array, img):
    Data = []
    for i in range(len(coordinate_array)):
        coord = coordinate_array[i]

        Data.append(img[coord[0], coord[1]])

    Data = np.array(Data)

    plt.scatter(Data[:, 1], Data[:, 2])
    plt.show()

    kmeans = KMeans(n_clusters=9)
    km = kmeans.fit(Data)
    km_labels = km.labels_

    plt.scatter(Data[:, 1], Data[:, 2],
                c=kmeans.labels_,
                s=70, cmap='Paired')
    plt.scatter(kmeans.cluster_centers_[:, 1],
                kmeans.cluster_centers_[:, 2],
                marker='^', s=100, linewidth=2,
                c=[0, 1, 2, 3, 4, 5, 6, 7, 8])

    plt.show()
    print(1)




if __name__ == '__main__':
    img = cv2.imread("../Data/PC/PaviaC(60,27,17).tif")
    coordinate_array = first_difference(img, 1, 10)
    img = cv2.imread("../Data/PC/PaviaC(60,27,17).tif")
    Cluster(coordinate_array, img)

    print(1)
