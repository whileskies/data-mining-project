import sys
sys.path.append("..")
from data_preprocessing import android_data_process as dp
from svm_classification import svm
import os
import numpy as np
import time


def class_labels_process(class_labels):
    new_class_labels = []
    for label in class_labels:
        if label == 0:
            new_class_labels.append(-1)
        else:
            new_class_labels.append(1)
    return new_class_labels


def classify():
    if not os.path.exists(dp.android_dataset_dir):
        print('数据预处理并保存中')
        android_dataset = dp.load_file_to_dataset()
        dp.save_android_dataset(android_dataset, dp.android_dataset_dir)
    else:
        android_dataset = dp.load_android_dataset(dp.android_dataset_dir)

    train_start_time = time.perf_counter()
    data = np.array(android_dataset.train_features)
    label = np.array(class_labels_process(android_dataset.train_class_labels))
    smo = svm.PlattSMO(data, label, 1, 0.0001, 10000, name='rbf', theta=20)
    smo.smoP()
    train_end_time = time.perf_counter()

    test_features = android_dataset.test_features
    test_class_labels = android_dataset.test_class_labels
    test_num = android_dataset.test_num

    predict_start_time = time.perf_counter()
    predict_labels = smo.predict(test_features)

    acc = 0
    for i in range(len(predict_labels)):
        class_label = predict_labels[i]
        if class_label == test_class_labels[i]:
            acc += 1
            print('正确', class_label)
        else:
            print('错误, 预测类别：%d, 正确类别：%d' % (class_label, test_class_labels[i]))

    print('\n正确率：%.2f%%' % (100.0 * acc / test_num))
    predict_end_time = time.perf_counter()

    print('SVM-训练阶段运行时间：%s秒' % (train_end_time - train_start_time))
    print('SVM-预测阶段运行时间：%s秒' % (predict_end_time - predict_start_time))


if __name__ == '__main__':
    classify()
