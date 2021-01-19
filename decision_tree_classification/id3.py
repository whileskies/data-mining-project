import math
import operator
import pickle


def calc_shannon_ent(data_set):
    num_entries = len(data_set)
    label_counts = {}
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1

    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * math.log(prob, 2)

    return shannon_ent


def split_data_set(data_set, axis, value):
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_data_set.append(reduced_feat_vec)

    return ret_data_set


def choose_best_feature_to_split(data_set):
    num_features = len(data_set[0]) - 1
    base_entropy = calc_shannon_ent(data_set)
    best_info_gain = 0.0
    best_feature = -1

    for i in range(num_features):
        # print(i)
        feat_list = [example[i] for example in data_set]
        unique_values = set(feat_list)
        new_entropy = 0.0
        for value in unique_values:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)

        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i

    return best_feature


def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    sub_labels = labels[:]
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]  # 当全部属于同一类时，停止划分
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)  # 当数据集没有更多的特征时，停止划分

    best_feat = choose_best_feature_to_split(data_set) # 选择增益最大的特征
    best_feat_label = labels[best_feat]
    d_tree = {best_feat_label: {}}
    del sub_labels[best_feat]

    feat_values = [example[best_feat] for example in data_set]
    unique_values = set(feat_values) # 当前结点样本每一类别创建一分支
    for value in unique_values:
        # 递归创建分支结点
        d_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)

    return d_tree


def classify(d_tree, feat_labels, test_vec):
    first_str = list(d_tree)[0]
    second_dict = d_tree[first_str]
    feat_index = feat_labels.index(first_str)
    key = test_vec[feat_index]
    if key not in second_dict:
        class_label = 0
    else:
        value_of_feat = second_dict[key]
        if isinstance(value_of_feat, dict):
            class_label = classify(value_of_feat, feat_labels, test_vec)
        else:
            class_label = value_of_feat

    return class_label


def store_tree(filename, d_tree, feature_labels, train_data_set, test_features_set, test_class_labels):
    with open(filename, 'wb') as f:
        pickle.dump(d_tree, f)
        pickle.dump(feature_labels, f)
        pickle.dump(train_data_set, f)
        pickle.dump(test_features_set, f)
        pickle.dump(test_class_labels, f)


def load_tree(filename):
    with open(filename, 'rb') as f:
        d_tree = pickle.load(f)
        feature_labels = pickle.load(f)
        train_data_set = pickle.load(f)
        test_features_set = pickle.load(f)
        test_class_labels = pickle.load(f)
    return d_tree, feature_labels, train_data_set, test_features_set, test_class_labels


def test():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    d_tree = create_tree(dataSet, labels)
    print(d_tree)


if __name__ == '__main__':
    test()
