import sys
sys.path.append("..")
import data_preprocessing.log_data_process as dp
from softmax_classification import softmax
import os
import time


def classify(classify_class1=True):
    if not os.path.exists(dp.bow_log_dataset_dir):
        print('数据预处理并保存中')
        log_dataset = dp.load_file_save_dataset()
    else:
        log_dataset = dp.load_log_dataset(dp.bow_log_dataset_dir)

    if classify_class1:
        test_features = log_dataset.test_features
        test_class_labels = log_dataset.test_class_labels1
        train_features = log_dataset.train_features
        train_class_labels = log_dataset.train_class_labels1
        label_map = log_dataset.class_label1_map
        class_num = log_dataset.class1_num
    else:
        test_features = log_dataset.test_features
        test_class_labels = log_dataset.test_class_labels2
        train_features = log_dataset.train_features
        train_class_labels = log_dataset.train_class_labels2
        label_map = log_dataset.class_label2_map
        class_num = log_dataset.class2_num

    train_start_time = time.perf_counter()
    softmax_clf = softmax.Softmax(train_features, train_class_labels,
                                  class_num, alpha=0.01, iterations=50)
    softmax_clf.train()
    train_end_time = time.perf_counter()

    predict_start_time = time.perf_counter()
    predict_class_labels = softmax_clf.predict(test_features)
    acc = 0
    for i in range(len(predict_class_labels)):
        class_label = predict_class_labels[i]
        actual_label = test_class_labels[i]
        if class_label == actual_label:
            acc += 1
            print('正确', label_map[class_label])
        else:
            print('错误, 预测类别：%s, 正确类别：%s' % (label_map[class_label], label_map[actual_label]))

    print('\n正确率：%.2f%%' % (100.0 * acc / log_dataset.test_num))
    predict_end_time = time.perf_counter()

    print('Softmax-训练阶段运行时间：%s秒' % (train_end_time - train_start_time))
    print('Softmax-预测阶段运行时间：%s秒' % (predict_end_time - predict_start_time))


if __name__ == '__main__':
    classify()
