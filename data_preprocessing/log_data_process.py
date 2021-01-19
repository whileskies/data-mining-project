import pickle
import random
import re

log_file_dir = '../data/Log/issues.csv'
stop_words_dir = '../data/Log/stop_words.txt'
bow_log_dataset_dir = '../data_preprocessing/pickle_data/log_bow_dataset.pickle'


class LogDataSet:
    def __init__(self, train_features, train_class_labels1, train_class_labels2, feature_labels, test_features,
                 test_class_labels1, test_class_labels2, class_label1_map, class_label2_map):
        self.train_features = train_features
        self.train_class_labels1 = train_class_labels1
        self.train_class_labels2 = train_class_labels2
        self.feature_labels = feature_labels
        self.test_features = test_features
        self.test_class_labels1 = test_class_labels1
        self.test_class_labels2 = test_class_labels2
        self.class_label1_map = class_label1_map
        self.class_label2_map = class_label2_map

        self.train_num = len(self.train_features)
        self.feature_num = len(self.train_features[0])
        self.test_num = len(self.test_features)

        self.class1_num = len(class_label1_map)
        self.class2_num = len(class_label2_map)

    def get_combined_train(self, is_label2=True):
        combined_train = []
        if is_label2:
            label = self.train_class_labels2
        else:
            label = self.train_class_labels1
        for i in range(self.train_num):
            tmp = self.train_features[i][:]
            tmp.append(label[i])
            combined_train.append(tmp)
        return combined_train


def read_log_file(filename):
    lines_list = []
    with open(filename) as f:
        for line in f.readlines():
            lines_list.append(line.strip())

    return lines_list


def read_stop_words(filename):
    stop_words = []
    with open(filename, encoding='UTF-8') as f:
        for line in f.readlines():
            stop_words.append(line.strip())

    return stop_words


def clean_log_file(lines):
    class_label1 = []
    class_label2 = []
    words_list = []

    for line in lines:
        last = line.rfind(',')
        last = line.rfind(',', last)
        line = line[:last]
        last = line.rfind(',')
        class_label2.append(line[last + 1:])
        line = line[:last]
        last = line.rfind(',')
        class_label1.append(line[last + 1:])
        line = line[:last]

        pattern = re.compile(r'[ ._\-=;@,?%"/\':()<>\[\]\\|\d#$&*]+')
        words = re.split(pattern, line)
        new_words = []
        for word in words:
            if len(word) == 0 or word in ['/', '\\']:
                continue
            word = word.lower()
            new_words.append(word)

        words_list.append(new_words)

    return words_list, class_label1, class_label2


def create_vocab_list(words_list, stop_words):
    vocab_set = set()
    for words in words_list:
        vocab_set = vocab_set | set(words)

    vocab_set = vocab_set - set(stop_words)
    return sorted(list(vocab_set))


def set_of_words2vec(vocab_list, words_list):
    sow = []
    for words in words_list:
        return_vec = [0] * len(vocab_list)
        for word in words:
            if word in vocab_list:
                return_vec[vocab_list.index(word)] = 1
        sow.append(return_vec)
    return sow


def bag_of_words2vec(vocab_list, words_list):
    bow = []
    for words in words_list:
        return_vec = [0] * len(vocab_list)
        for word in words:
            if word in vocab_list:
                return_vec[vocab_list.index(word)] += 1
        bow.append(return_vec)
    return bow


def encode_label(label_list):
    encoded_label = []
    label_map = sorted(list(set(label_list)))
    for label in label_list:
        encoded_label.append(label_map.index(label))
    return label_map, encoded_label


def load_file_to_dataset(is_bow=True, train_dataset_rate=0.8):
    lines = read_log_file(log_file_dir)
    stop_words = read_stop_words(stop_words_dir)
    words_list, class_label1, class_label2 = clean_log_file(lines)
    vocab_list = create_vocab_list(words_list, stop_words)
    if is_bow:
        features = bag_of_words2vec(vocab_list, words_list)
    else:
        features = set_of_words2vec(vocab_list, words_list)

    class_label1_map, class_label1_num = encode_label(class_label1)
    class_label2_map, class_label2_num, = encode_label(class_label2)

    random_index = [i for i in range(len(features))]
    random.shuffle(random_index)

    train_features = []
    train_class_labels1 = []
    train_class_labels2 = []
    test_features = []
    test_class_labels1 = []
    test_class_labels2 = []
    cnt = 0
    train_total_cnt = int(len(features) * train_dataset_rate)
    for i in range(len(random_index)):
        if cnt < train_total_cnt:
            train_features.append(features[random_index[i]])
            train_class_labels1.append(class_label1_num[random_index[i]])
            train_class_labels2.append(class_label2_num[random_index[i]])
        else:
            test_features.append(features[random_index[i]])
            test_class_labels1.append(class_label1_num[random_index[i]])
            test_class_labels2.append(class_label2_num[random_index[i]])
        cnt += 1

    return LogDataSet(train_features, train_class_labels1, train_class_labels2, vocab_list, test_features,
                      test_class_labels1, test_class_labels2, class_label1_map, class_label2_map)


def save_log_dataset(log_dataset, filename):
    with open(filename, 'wb') as f:
        pickle.dump(log_dataset, f)


def load_log_dataset(filename):
    with open(filename, 'rb') as f:
        log_dataset = pickle.load(f)
    return log_dataset


def load_file_save_dataset():
    log_dataset = load_file_to_dataset(train_dataset_rate=0.8)
    save_log_dataset(log_dataset, bow_log_dataset_dir)
    return log_dataset





