import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, cohen_kappa_score, f1_score


import linecache
import cv2
def Get_intrinal_sample(Path):
    Init_label = []
    Init_coor = []
    for j in range(1, len(open(Path).readlines()) + 1):
        lindata = linecache.getline(Path, j)
        data_label,data_x, data_y = int(lindata.split(',')[0]), int(lindata.split(',')[1]), int(lindata.split(',')[2])
        Init_coor.append([data_x, data_y])
        Init_label.append(data_label)
    return Init_coor,Init_label


def Evaluation(coordinates, y_true, image_data, labels):

    # 获取影像中对应坐标位置的标签
    predicted_labels = []
    for coord in coordinates:
        x, y = coord[0],coord[1]
        label = image_data[x,y][0]
        predicted_labels.append(label)

    y_pred = np.array(predicted_labels)
    """
     计算多分类评价指标。

     :param y_true: 真实标签数组
     :param y_pred: 预测标签数组
     :param labels: 标签列表
     :return: 指标字典
     """
    metrics = {}

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics['Confusion Matrix'] = cm

    # Overall Accuracy (OA)
    oa = accuracy_score(y_true, y_pred)
    metrics['Overall Accuracy (OA)'] = oa

    # Average Accuracy (AA)
    aa = np.mean(np.diag(cm) / np.sum(cm, axis=1))
    metrics['Average Accuracy (AA)'] = aa

    # Kappa Coefficient (Ka)
    ka = cohen_kappa_score(y_true, y_pred)
    metrics['Kappa Coefficient (Ka)'] = ka

    # F1-Score
    f1 = f1_score(y_true, y_pred, average='weighted')
    metrics['F1-Score'] = f1

    # User Accuracy (Precision for each class)
    class_report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    user_accuracy = {label: class_report[str(label)]['precision'] for label in labels}
    metrics['User Accuracy'] = user_accuracy

    return metrics




Dates = ['201812', '201901', '201902','201903','201904','201905','201906','201907','201908','201910','201911','201912',
              '202001','202002','202003','202004','202005','202006','202007','202008','202009','202010','202011','202012',
              '202101','202102','202103','202104','202105','202106','202107','202108','202109','202110','202111','202112',
              '202201','202202','202203','202204','202205','202206','202207','202208','202209','202210','202211']

f = open("..\Data\WuHan\Intermediate results\SVM\\45\Result-4000/result.txt", 'a')

for date in Dates:
    f.write('-------------------------------------------------------------------------------------------------------------'+'\n')
    label = [1,2,3,4,5,6]
    Init_coor, Init_label = Get_intrinal_sample("../Data/WuHan/verfication/WHU_{0}.txt".format(date))
    image_data = cv2.imread("..\Data\WuHan\Intermediate results\SVM\\45\Result-4000/{0}_initial_result.bmp".format(date))
    Init_coor = np.array(Init_coor)
    Init_label = np.array(Init_label)
    metric = Evaluation(Init_coor, Init_label, image_data, label)
    for key, value in metric.items():
        f.write(f"{key}:{value}"+'\n')


    image_data1 = cv2.imread("..\Data\WuHan\Intermediate results\SVM\\45\Result-4000//{0}_sampling_result.bmp".format(date))
    Init_coor = np.array(Init_coor)
    Init_label = np.array(Init_label)
    metric1 = Evaluation(Init_coor, Init_label, image_data1, label)
    for key, value in metric1.items():
        f.write(f"{key}:{value}"+'\n')





