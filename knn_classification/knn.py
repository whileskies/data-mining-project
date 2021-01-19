import numpy as np
import collections


def classify(test_feature, train_features, train_labels, k):
    test_feature = np.array(test_feature)
    train_features = np.array(train_features)

    # 计算距离
    dist = np.sum((test_feature - train_features) ** 2, axis=1) ** 0.5
    # k个最近的标签
    k_labels = [train_labels[index] for index in dist.argsort()[0: k]]
    # 出现次数最多的标签即为最终类别
    label = collections.Counter(k_labels).most_common(1)[0][0]
    return label
