from PIL import Image
import os
import sys
import numpy as np
import time
from sklearn import svm
import joblib


# 获取指定路径下的所有 .png 文件
def get_file_list(path):
    # file_list = []
    # for filename in os.listdir(path):
    #     ele_path = os.path.join(path, filename)
    #     for imgname in os.listdir(ele_path):
    #         subele_path = os.path.join(ele_path, imgname)
    #         if (subele_path.endswith(".png")):
    #             file_list.append(subele_path)
    # return file_list
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".png")]


# 解析出 .png 图件文件的名称
def get_img_name_str(imgPath):
    return imgPath.split(os.path.sep)[-1]


# 将 20px * 20px 的图像数据转换成 1*400 的 numpy 向量
# 参数：imgFile--图像名  如：0_1.png
# 返回：1*400 的 numpy 向量
def img2vector(imgFile):
    # print("in img2vector func--para:{}".format(imgFile))
    img = Image.open(imgFile).convert('L')
    img_arr = np.array(img, 'i')  # 20px * 20px 灰度图像
    img_normalization = np.round(img_arr / 255)  # 对灰度值进行归一化
    img_arr2 = np.reshape(img_normalization, (1, -1))  # 1 * 400 矩阵
    return img_arr2


# 读取一个类别的所有数据并转换成矩阵
# 参数：
#    basePath: 图像数据所在的基本路径
#       Mnist-image/train/
#       Mnist-image/test/
#    cla：类别名称
#       0,1,2,...,9
# 返回：某一类别的所有数据----[样本数量*(图像宽x图像高)] 矩阵
def read_and_convert(imgFileList):
    dataLabel = []  # 存放类标签
    dataNum = len(imgFileList)
    dataMat = np.zeros((dataNum, 400))  # dataNum * 400 的矩阵
    for i in range(dataNum):
        imgNameStr = imgFileList[i]
        imgName = get_img_name_str(imgNameStr)  # 得到 数字_实例编号.png
        # print("imgName: {}".format(imgName))
        classTag = imgName.split(".")[0].split("_")[0]  # 得到 类标签(数字)
        # print("classTag: {}".format(classTag))
        dataLabel.append(classTag)
        dataMat[i, :] = img2vector(imgNameStr)
    return dataMat, dataLabel


# 读取训练数据
def read_all_data():
    cName = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    path = sys.path[1]
    train_data_path = os.path.join(path, 'data\\Mnist-image\\train\\0')
    # print(train_data_path)
    # train_data_path = "Mnist-image\\train\\0"
    print('0')
    flist = get_file_list(train_data_path)
    dataMat, dataLabel = read_and_convert(flist)
    for c in cName:
        print(c)
        train_data_path = os.path.join(path, 'data\\Mnist-image\\train\\') + c
        flist_ = get_file_list(train_data_path)
        dataMat_, dataLabel_ = read_and_convert(flist_)
        dataMat = np.concatenate((dataMat, dataMat_), axis=0)
        dataLabel = np.concatenate((dataLabel, dataLabel_), axis=0)
    # print(dataMat.shape)
    # print(len(dataLabel))
    return dataMat, dataLabel


# create model
def create_svm(dataMat, dataLabel, path, decision='ovr'):
    clf = svm.SVC(C=1.0, kernel='rbf', decision_function_shape=decision)
    rf = clf.fit(dataMat, dataLabel)
    joblib.dump(rf, path)
    return clf


'''
SVC参数
svm.SVC(C=1.0,kernel='rbf',degree=3,gamma='auto',coef0=0.0,shrinking=True,probability=False,
tol=0.001,cache_size=200,class_weight=None,verbose=False,max_iter=-1,decision_function_shape='ovr',random_state=None)
C：C-SVC的惩罚参数C?默认值是1.0
C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时
准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。
kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 
       0 – 线性：u'v
 　　 1 – 多项式：(gamma*u'*v + coef0)^degree
  　　2 – RBF函数：exp(-gamma|u-v|^2)
  　　3 –sigmoid：tanh(gamma*u'*v + coef0)
degree ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。（没用）
gamma ： ‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1/n_features
coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。（没用）
probability ：是否采用概率估计？.默认为False
shrinking ：是否采用shrinking heuristic方法，默认为true
tol ：停止训练的误差值大小，默认为1e-3
cache_size ：核函数cache缓存大小，默认为200
class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight*C(C-SVC中的C)
verbose ：允许冗余输出？
max_iter ：最大迭代次数。-1为无限制。
decision_function_shape ：‘ovo’, ‘ovr’ or None, default=None3（选用ovr，一对多）
random_state ：数据洗牌时的种子值，int值
主要调节的参数有：C、kernel、degree、gamma、coef0
'''

if __name__ == '__main__':
    # clf = svm.SVC(decision_function_shape='ovr')
    st = start = time.perf_counter()
    dataMat, dataLabel = read_all_data()
    path = sys.path[1]
    model_path = os.path.join(path, 'model\\svm.model')
    create_svm(dataMat, dataLabel, model_path, decision='ovr')
    et = time.perf_counter()
    print("Training spent {:.4f}s.".format((et - st)))
