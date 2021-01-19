import sys
sys.path.append("..")
from data_preprocessing import android_data_process as dp
from knn_classification import knn
import os
import time


def classify():
    if not os.path.exists(dp.android_dataset_dir):
        print('数据预处理并保存中')
        android_dataset = dp.load_file_to_dataset()
        dp.save_android_dataset(android_dataset, dp.android_dataset_dir)
    else:
        android_dataset = dp.load_android_dataset(dp.android_dataset_dir)

    test_features = android_dataset.test_features
    test_class_labels = android_dataset.test_class_labels
    test_num = android_dataset.test_num
    train_features = android_dataset.train_features
    train_class_labels = android_dataset.train_class_labels

    predict_start_time = time.perf_counter()
    acc = 0
    for i in range(test_num):
        class_label = knn.classify(test_features[i], train_features, train_class_labels, 7)
        if class_label == test_class_labels[i]:
            acc += 1
            print('正确', class_label)
        else:
            print('错误, 预测类别：%d, 正确类别：%d' % (class_label, test_class_labels[i]))

    print('\n正确率：%.2f%%' % (100.0 * acc / test_num))
    predict_end_time = time.perf_counter()
    print('KNN-预测阶段运行时间：%s秒' % (predict_end_time - predict_start_time))


if __name__ == '__main__':
    classify()

