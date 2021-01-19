import numpy as np


def train_nb(train_matrix, train_category, classes_num):
    p_class = [0] * classes_num
    for i in range(len(p_class)):
        # 计算每个类别的概率，并做laplace平滑
        p_class[i] = np.log((train_category.count(i) + 1.0) / (len(train_category) + classes_num))

    num_words = len(train_matrix[0])
    p_words_num = []
    p_words_denom = []
    p_words = []

    for i in range(classes_num):
        # laplace平滑
        p_words_num.append(np.ones(num_words))
        p_words_denom.append(num_words)

    for i in range(len(train_matrix)):
        # 计算每个类别下每个特征出现的总次数
        p_words_num[train_category[i]] += train_matrix[i]
        # 计算每个类别下出现的特征总次数
        p_words_denom[train_category[i]] += np.sum(train_matrix[i])

    for i in range(classes_num):
        # 计算p(xi|y)，每个特征在一类别下出现的概率
        p_words.append(np.log(p_words_num[i] / p_words_denom[i]))

    return p_words, p_class


def nb_classify(p_words, p_class, test_features, classes_num):
    probability = []

    for class_index in range(classes_num):
        log_sum = p_class[class_index]
        # 计算特征向量属于class_index类别的概率
        for i in range(len(test_features)):
            if test_features[i] > 0:
                log_sum += test_features[i] * p_words[class_index][i]

        probability.append(log_sum)

    # 选择概率最大的类别
    return np.argmax(probability)

