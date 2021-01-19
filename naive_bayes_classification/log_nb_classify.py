import sys
sys.path.append("..")
from data_preprocessing import log_data_process as dp
from naive_bayes_classification import naive_bayes as nb
import os
import time


def classify(classify_class1=True):
    if not os.path.exists(dp.bow_log_dataset_dir):
        print('数据预处理并保存中')
        log_dataset = dp.load_file_save_dataset()
    else:
        log_dataset = dp.load_log_dataset(dp.bow_log_dataset_dir)

    test_features = log_dataset.test_features
    test_num = log_dataset.test_num

    if classify_class1:
        train_class_labels = log_dataset.train_class_labels1
        test_class_labels = log_dataset.test_class_labels1
        class_num = log_dataset.class1_num
        label_map = log_dataset.class_label1_map
    else:
        train_class_labels = log_dataset.train_class_labels2
        test_class_labels = log_dataset.test_class_labels2
        class_num = log_dataset.class2_num
        label_map = log_dataset.class_label2_map

    train_start_time = time.perf_counter()
    p_words, p_class = nb.train_nb(log_dataset.train_features, train_class_labels, class_num)
    train_end_time = time.perf_counter()

    predict_start_time = time.perf_counter()
    acc = 0
    for i in range(test_num):
        class_label = nb.nb_classify(p_words, p_class, test_features[i], class_num)
        if class_label == test_class_labels[i]:
            acc += 1
            print('正确', label_map[class_label])
        else:
            print('错误, 预测类别：%s, 正确类别：%s' % (label_map[class_label], label_map[test_class_labels[i]]))

    print('\n正确率：%.2f%%' % (100.0 * acc / test_num))
    predict_end_time = time.perf_counter()

    print('Naive Bayes-训练阶段运行时间：%s秒' % (train_end_time - train_start_time))
    print('Naive Bayes-预测阶段运行时间：%s秒' % (predict_end_time - predict_start_time))


if __name__ == '__main__':
    classify()
