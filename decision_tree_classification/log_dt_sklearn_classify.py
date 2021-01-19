import sys
sys.path.append("..")
import data_preprocessing.log_data_process as dp
from sklearn import tree
import os
import time


def classify(classify_class1=True):
    if not os.path.exists(dp.bow_log_dataset_dir):
        print('数据预处理并保存中')
        log_dataset = dp.load_file_save_dataset()
    else:
        log_dataset = dp.load_log_dataset(dp.bow_log_dataset_dir)

    if classify_class1:
        train_class_labels = log_dataset.train_class_labels1
        label_map = log_dataset.class_label1_map
        test_class_labels = log_dataset.test_class_labels1
    else:
        train_class_labels = log_dataset.train_class_labels2
        label_map = log_dataset.class_label2_map
        test_class_labels = log_dataset.test_class_labels2

    d_tree = tree.DecisionTreeClassifier()
    build_tree_start_time = time.perf_counter()
    d_tree.fit(log_dataset.train_features, train_class_labels)
    build_tree_end_time = time.perf_counter()
    print(tree.export_text(d_tree, feature_names=log_dataset.feature_labels))

    predict_start_time = time.perf_counter()
    acc = 0
    for i in range(log_dataset.test_num):
        class_label = d_tree.predict([log_dataset.test_features[i]])[0]
        actual_label = test_class_labels[i]
        if class_label == actual_label:
            acc += 1
            print('正确', label_map[class_label])
        else:
            print('错误, 预测类别：%s, 正确类别：%s' % (label_map[class_label], label_map[actual_label]))

    print('\n正确率：%.2f%%' % (100.0 * acc / log_dataset.test_num))
    predict_end_time = time.perf_counter()

    print('决策树-sklearn-构建决策树运行时间：%s秒' % (build_tree_end_time - build_tree_start_time))
    print('决策树-sklearn-预测阶段运行时间：%s秒' % (predict_end_time - predict_start_time))


if __name__ == '__main__':
    classify()
