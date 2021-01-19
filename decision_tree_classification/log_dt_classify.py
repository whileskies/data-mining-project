import sys
sys.path.append("..")
from data_preprocessing import log_data_process as dp
from decision_tree_classification import id3


def classify():
    log_dataset = dp.load_log_dataset(dp.bow_log_dataset_dir)
    train_data = log_dataset.get_combined_train()
    d_tree = id3.create_tree(train_data, log_dataset.feature_labels)
    print(d_tree)


if __name__ == '__main__':
    classify()
