import sys
import time
import svm
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt


def svmtest(model_path):
    path = sys.path[1]
    tbasePath = os.path.join(path, "data\\Mnist-image\\test\\")
    tcName = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    tst = time.perf_counter()
    allErrCount = 0
    allErrorRate = 0.0
    allScore = 0.0
    ErrCount = np.zeros(10, int)
    TrueCount = np.zeros(10, int)
    # 加载模型
    clf = joblib.load(model_path)
    for tcn in tcName:
        testPath = tbasePath + tcn
        # print("class " + tcn + " path is: {}.".format(testPath))
        tflist = svm.get_file_list(testPath)
        # tflist
        tdataMat, tdataLabel = svm.read_and_convert(tflist)
        print("test dataMat shape: {0}, test dataLabel len: {1} ".format(tdataMat.shape, len(tdataLabel)))
        # print("test dataLabel: {}".format(len(tdataLabel)))
        pre_st = time.perf_counter()
        preResult = clf.predict(tdataMat)
        pre_et = time.perf_counter()
        print("Recognition  " + tcn + " spent {:.4f}s.".format((pre_et - pre_st)))
        # print("predict result: {}".format(len(preResult)))
        errCount = len([x for x in preResult if x != tcn])
        ErrCount[int(tcn)] = errCount
        TrueCount[int(tcn)] = len(tdataLabel) - errCount
        print("errorCount: {}.".format(errCount))
        allErrCount += errCount
        score_st = time.perf_counter()
        score = clf.score(tdataMat, tdataLabel)
        score_et = time.perf_counter()
        print("computing score spent {:.6f}s.".format(score_et - score_st))
        allScore += score
        print("score: {:.6f}.".format(score))
        print("error rate is {:.6f}.".format((1 - score)))

    tet = time.perf_counter()
    print("Testing All class total spent {:.6f}s.".format(tet - tst))
    print("All error Count is: {}.".format(allErrCount))
    avgAccuracy = allScore / 10.0
    print("Average accuracy is: {:.6f}.".format(avgAccuracy))
    print("Average error rate is: {:.6f}.".format(1 - avgAccuracy))
    print("number", " TrueCount", " ErrCount")
    for tcn in tcName:
        tcn = int(tcn)
        print(tcn, "     ", TrueCount[tcn], "      ", ErrCount[tcn])
    plt.figure(figsize=(12, 6))
    x = list(range(10))
    plt.plot(x, TrueCount, color='blue', label="TrueCount")  # 将正确的数量设置为蓝色
    plt.plot(x, ErrCount, color='red', label="ErrCount")  # 将错误的数量为红色
    plt.legend(loc='best')  # 显示图例的位置，这里为右下方
    plt.title('Projects')
    plt.xlabel('number')  # x轴标签
    plt.ylabel('count')  # y轴标签
    plt.xticks(np.arange(10), ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    plt.show()


if __name__ == '__main__':
    path = sys.path[1]
    model_path = os.path.join(path, 'model\\svm.model')
    svmtest(model_path)

