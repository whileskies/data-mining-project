import sys
sys.path.append("..")
from data_preprocessing import android_data_process as dp
from decision_tree_classification import id3
import os
import time

tree_file_dir = 'pickle_data/android_dt_tree.pickle'


def classify():
    build_tree_start_time = 0
    build_tree_end_time = 0
    if os.path.exists(tree_file_dir):
        print('决策树已保存')
        d_tree, feature_labels, train_data_set, test_features_set, test_class_labels = id3.load_tree(tree_file_dir)
    else:
        print('决策树未保存，重新建树中')
        if not os.path.exists(dp.android_dataset_dir):
            print('数据预处理并保存中')
            android_dataset = dp.load_file_save_dataset()
        else:
            android_dataset = dp.load_android_dataset(dp.android_dataset_dir)
        train_data_set = android_dataset.get_combined_train()
        feature_labels = android_dataset.feature_labels
        test_features_set = android_dataset.test_features
        test_class_labels = android_dataset.test_class_labels
        # print(feature_labels)

        build_tree_start_time = time.perf_counter()
        d_tree = id3.create_tree(train_data_set, feature_labels)
        build_tree_end_time = time.perf_counter()
        id3.store_tree(tree_file_dir, d_tree, feature_labels, train_data_set, test_features_set, test_class_labels)

    print(d_tree)

    predict_start_time = time.perf_counter()
    acc = 0
    for i in range(len(test_features_set)):
        class_label = id3.classify(d_tree, feature_labels, test_features_set[i])
        if class_label == test_class_labels[i]:
            acc += 1
            print('正确', class_label)
        else:
            print('错误, 预测类别：%d, 正确类别：%d' % (class_label, test_class_labels[i]))

    print('\n正确率：%.2f%%' % (100.0 * acc / len(test_features_set)))
    predict_end_time = time.perf_counter()

    print('决策树-构建决策树运行时间：%s秒' % (build_tree_end_time - build_tree_start_time))
    print('决策树-预测阶段运行时间：%s秒' % (predict_end_time - predict_start_time))


if __name__ == '__main__':
    classify()
