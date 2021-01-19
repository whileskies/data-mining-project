from numpy import *


def selectJrand(i, m):
    j = i  # 选择一个不等于i的j
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


class PlattSMO:
    def __init__(self, dataMat, classlabels, C, toler, maxIter, **kernelargs):
        """
        构造函数
        :param dataMat: 特征向量，二维数组
        :param classlabels: 数据标签，一维数组
        :param C: 对松弛变量的容忍程度，越大越不容忍
        :param toler: 完成一次迭代维度误差要求
        :param maxIter: 迭代次数
        :param kernelargs: 核参数，特别注意高斯核会有两个参数，但是都是通过这一个变量传进来的
        """
        self.x = array(dataMat)
        self.label = array(classlabels).transpose()
        self.C = C
        self.toler = toler
        self.maxIter = maxIter
        self.m = shape(dataMat)[0]  # 输入的行数，也即是表示了有多少个输入
        self.n = shape(dataMat)[1]  # 列数，表示每一个输入有多少个特征向量
        self.alpha = array(zeros(self.m), dtype='float64')  # 初始化alpha
        self.b = 0.0
        self.eCache = array(zeros((self.m, 2)))  # 错误缓存
        self.K = zeros((self.m, self.m), dtype='float64')  # 先求内积，
        self.kwargs = kernelargs
        self.SV = ()  # 最后保留的支持向量
        self.SVIndex = None  # 支持向量的索引

        for i in range(self.m):
            for j in range(self.m):
                self.K[i, j] = self.kernelTrans(self.x[i, :], self.x[j, :])

    def calcEK(self, k):
        """
        计算第k个数据的误差
        :param k:
        :return:
        """
        # 因为这是训练阶段，用数据集的数据，所以可以直接这样做，这里先把内积全部求了，
        fxk = dot(self.alpha * self.label, self.K[:, k]) + self.b
        Ek = fxk - float(self.label[k])
        return Ek

    def updateEK(self, k):
        Ek = self.calcEK(k)

        self.eCache[k] = [1, Ek]

    def selectJ(self, i, Ei):
        """
        在确定了一个参数的前提下，按照最大步长取另一个参数
        :param i:
        :param Ei:
        :return:
        """
        maxE = 0.0
        selectJ = 0
        Ej = 0.0
        validECacheList = nonzero(self.eCache[:, 0])[0]
        if len(validECacheList) > 1:
            for k in validECacheList:
                if k == i: continue
                Ek = self.calcEK(k)
                deltaE = abs(Ei - Ek)
                if deltaE > maxE:
                    selectJ = k
                    maxE = deltaE
                    Ej = Ek
            return selectJ, Ej
        else:
            selectJ = selectJrand(i, self.m)
            Ej = self.calcEK(selectJ)
            return selectJ, Ej

    def innerL(self, i):
        """
        选择参数之后，更新参数
        """
        Ei = self.calcEK(i)

        if (self.label[i] * Ei < -self.toler and self.alpha[i] < self.C) or \
                (self.label[i] * Ei > self.toler and self.alpha[i] > 0):

            self.updateEK(i)

            j, Ej = self.selectJ(i, Ei)

            alphaIOld = self.alpha[i].copy()
            alphaJOld = self.alpha[j].copy()

            if self.label[i] != self.label[j]:
                L = max(0, self.alpha[j] - self.alpha[i])
                H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
            else:
                L = max(0, self.alpha[j] + self.alpha[i] - self.C)
                H = min(self.C, self.alpha[i] + self.alpha[j])
            if L == H:
                return 0

            eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
            if eta >= 0:
                return 0

            self.alpha[j] -= self.label[j] * (Ei - Ej) / eta
            self.alpha[j] = clipAlpha(self.alpha[j], H, L)
            self.updateEK(j)

            if abs(alphaJOld - self.alpha[j]) < 0.00001:  # 目标迭代完成，不需要再迭代，而且所有参数已经保留，理论上是不应该保留当前的参数的，但是也正因为反正相差不多，所以可以
                return 0

            self.alpha[i] += self.label[i] * self.label[j] * (alphaJOld - self.alpha[j])
            self.updateEK(i)

            b1 = self.b - Ei - self.label[i] * self.K[i, i] * (self.alpha[i] - alphaIOld) - \
                 self.label[j] * self.K[i, j] * (self.alpha[j] - alphaJOld)
            b2 = self.b - Ej - self.label[i] * self.K[i, j] * (self.alpha[i] - alphaIOld) - \
                 self.label[j] * self.K[j, j] * (self.alpha[j] - alphaJOld)
            if 0 < self.alpha[i] and self.alpha[i] < self.C:
                self.b = b1
            elif 0 < self.alpha[j] and self.alpha[j] < self.C:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0

    def smoP(self):
        """
        外层大循环，
        :return:
        """
        iter = 0
        entrySet = True
        alphaPairChanged = 0
        while iter < self.maxIter and ((alphaPairChanged > 0) or (entrySet)):
            alphaPairChanged = 0
            if entrySet:
                for i in range(self.m):
                    alphaPairChanged += self.innerL(i)
                iter += 1
            else:
                nonBounds = nonzero((self.alpha > 0) * (self.alpha < self.C))[0]
                for i in nonBounds:
                    alphaPairChanged += self.innerL(i)
                iter += 1
            if entrySet:
                entrySet = False
            elif alphaPairChanged == 0:
                entrySet = True
        # 保存模型参数
        self.SVIndex = nonzero(self.alpha)[0]  # 取非0部分把应该
        self.SV = self.x[self.SVIndex]
        self.SVAlpha = self.alpha[self.SVIndex]
        self.SVLabel = self.label[self.SVIndex]

        # 清空中间变量
        self.x = None
        self.K = None
        self.label = None
        self.alpha = None
        self.eCache = None

    def kernelTrans(self, x, z):
        """
        核函数说到底就是求内积，求输入x和已有数据标签的内积，最后的结果是一个常数
        内积有两个，一个是向量形式的对应相乘相加一个是矩阵形式的矩阵乘积
        输入是两个要求内积的一维数组，也就是说要拆成一个个来做
        :param x:
        :param z:
        :return:
        """
        if array(x).ndim != 1 or array(x).ndim != 1:
            raise Exception("input vector is not 1 dim")
        if self.kwargs['name'] == 'linear':
            return sum(x * z)
        elif self.kwargs['name'] == 'rbf':
            theta = self.kwargs['theta']
            return exp(sum((x - z) * (x - z)) / (-1 * theta ** 2))

    def calcw(self):
        """
        计算w，结果是一个数组，长度是特征向量的长度
        :return:
        """
        for i in range(self.m):
            self.w += dot(self.alpha[i] * self.label[i], self.x[i, :])

    def predict(self, testData):
        """
        输入待预测的数据，输出结果
        :param testData: 待预测数据，要是数组形式，二维数组
        :return: 一个列表，包含结果
        """
        test = array(testData)
        # return (test * self.w + self.b).getA()
        result = []
        m = shape(test)[0]
        for i in range(m):
            tmp = self.b
            for j in range(len(self.SVIndex)):
                # wx+b,w和SVAlpha，Label以及核函数有关
                # 求和可以是直接求和，也可以转成矩阵，这里就是直接求和
                # 计算支持向量的数目，用他们来进行估计
                tmp += self.SVAlpha[j] * self.SVLabel[j] * self.kernelTrans(self.SV[j], test[i, :])

            # 不可分的情况下，其实就可以，直接指定想要的情况，如果需要的话，工程中会有这个要求
            while tmp == 0:
                tmp = random.uniform(-1, 1)
            if tmp > 0:
                tmp = 1
            else:
                # tmp = -1
                tmp = 0
            result.append(tmp)
        return result
