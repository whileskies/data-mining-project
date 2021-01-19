# classification
This is a classification task of data mining course

### 任务

实现不少于四个分类算法，至少包括决策树算法、KNN、SVM，并需要在安卓恶意软件检测数据集和日志分析数据集上进行测试

### 数据集

##### 安卓恶意软件检测

安卓恶意软件检测数据集直接是已经提取好的特征向量，包含 1500 个样本，特征向量为 887 维，前 209 维对应着 209 个敏感 api，后面 174 维对应着 174 类常用 Intent，然后138 维表示 permission，接着 365 维是包特征，最后一维是类别，0 代表良性，1 代表恶意

##### 日志分类数据集

日志数据集是文本数据，同样使用 csv 格式存储，以其中一行为例：

```shell
.exe and missing folders; NA, OS / SYSTEM SOFTWARE, OS / SYSTEM SOFTWARE|REBOOT, 91313904
```

第一列为日志信息，为一串英文的计算机日志，第二列为该日志所属的大类别，第三列为日志所属于的小类别，也即大类别下的子类别

### 使用

- 本项目采用python 3，部分代码需要numpy, sklearn第三方库

- 以运行决策树分类算法为例：

1. 进入decision_tree_classification目录

```shell
cd decision_tree_classification
```

2. 运行安卓数据集分类

```shell
python android_dt_classify.py
```

3. 运行日志数据集分类，需要sklearn

```shell
python log_dt_sklearn_classify.py
```

4. 退出该目录

```shell
cd ..
```

- 运行日志数据集小分类时，只需将以log开头的代码classify函数指定False参数即可：

```python
# 日志数据集分大类
if __name__ == '__main__':
    classify()
    
# 日志数据集分小类
if __name__ == '__main__':
    classify(False)
```

- data_preprocessing目录为数据预处理的代码，预处理完成后使用pickle序列化以文件方式保存在data_preprocessing\pickle_data目录，如果该目录下不含android_dataset.pickle、log_bow_dataset.pickle文件，运行任一分类代码时首先会进行重新生成，因此这两个文件可以删除以减少代码体积