import sys
sys.path.append("..")
import data_preprocessing.log_data_process as dp
from knn_classification import knn
import os


def classify(classify_class1=True):
    if not os.path.exists(dp.bow_log_dataset_dir):
        log_dataset = dp.load_file_save_dataset()
    else:
        log_dataset = dp.load_log_dataset(dp.bow_log_dataset_dir)

    if classify_class1:
        test_features = log_dataset.test_features
        test_class_labels = log_dataset.test_class_labels1
        train_features = log_dataset.train_features
        train_class_labels = log_dataset.train_class_labels1
        label_map = log_dataset.class_label1_map
    else:
        test_features = log_dataset.test_features
        test_class_labels = log_dataset.test_class_labels2
        train_features = log_dataset.train_features
        train_class_labels = log_dataset.train_class_labels2
        label_map = log_dataset.class_label2_map

    acc = 0
    for i in range(log_dataset.test_num):
        class_label = knn.classify(test_features[i], train_features, train_class_labels, 3)
        actual_label = test_class_labels[i]
        if class_label == actual_label:
            acc += 1
            print('正确', class_label)
        else:
            print('错误, 预测类别：%s, 正确类别：%s' % (label_map[class_label], label_map[actual_label]))

    print('正确率：%.2f%%' % (100.0 * acc / log_dataset.test_num))


if __name__ == '__main__':
    classify()
