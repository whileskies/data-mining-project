import random
import pickle

file_dir = '../data/Android Malware Detection Data/Android Malware Detection Data.csv'
android_dataset_dir = '../data_preprocessing/pickle_data/android_dataset.pickle'


class AndroidDataSet:
    def __init__(self, train_features, train_class_labels, feature_labels, test_features, test_class_labels):
        self.train_features = train_features
        self.train_class_labels = train_class_labels
        self.feature_labels = feature_labels
        self.test_features = test_features
        self.test_class_labels = test_class_labels

        self.train_num = len(self.train_features)
        self.feature_num = len(self.train_features[0])
        self.test_num = len(self.test_features)
        self.class_num = 2

    def get_combined_train(self):
        combined_train = []

        for i in range(self.train_num):
            tmp = self.train_features[i][:]
            tmp.append(self.train_class_labels[i])
            combined_train.append(tmp)
        return combined_train


def get_data_from_file():
    features = []
    class_labels = []

    with open(file_dir) as f:
        for line in f.readlines():
            line_list = list(map(lambda x: int(x), line.strip().split(',')))
            features.append(line_list[:len(line_list) - 1])
            class_labels.append(line_list[-1])

    feature_labels = [str(i) for i in range(len(features[0]))]

    return features, class_labels, feature_labels


def load_file_to_dataset(train_dataset_rate=0.8):
    features, class_labels, feature_labels = get_data_from_file()

    train_features = []
    train_class_labels = []
    test_features = []
    test_class_labels = []

    cnt = 0
    train_total_cnt = int(len(features) * train_dataset_rate)

    random_index = [i for i in range(len(features))]
    random.shuffle(random_index)

    for i in range(len(random_index)):
        if cnt < train_total_cnt:
            train_features.append(features[random_index[i]])
            train_class_labels.append(class_labels[random_index[i]])
        else:
            test_features.append(features[random_index[i]])
            test_class_labels.append(class_labels[random_index[i]])
        cnt += 1

    return AndroidDataSet(train_features, train_class_labels, feature_labels, test_features, test_class_labels)


def save_android_dataset(android_dataset, filename):
    with open(filename, 'wb') as f:
        pickle.dump(android_dataset, f)


def load_android_dataset(filename):
    with open(filename, 'rb') as f:
        android_dataset = pickle.load(f)
    return android_dataset


def load_file_save_dataset():
    android_dataset = load_file_to_dataset()
    save_android_dataset(android_dataset, android_dataset_dir)
    return android_dataset


