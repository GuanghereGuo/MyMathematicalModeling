import numpy as np


class EntropyWeightMethod:
    def __init__(self, X):
        """初始化类，接受指标矩阵 X"""
        self.X = np.array(X)  # 将输入转换为 numpy 数组
        self.Z = None  # 标准化矩阵，初始为 None
        self.W = None  # 权重，初始为 None

    @staticmethod
    def mylog(p):
        """自定义对数函数，处理零元素"""
        n = len(p)
        lnp = np.zeros(n)
        for i in range(n):
            if p[i] == 0:
                lnp[i] = 0  # 对零返回 0
            else:
                lnp[i] = np.log(p[i])  # 计算自然对数
        return lnp

    def standardize(self):
        """计算标准化矩阵 Z"""
        self.Z = self.X / np.sqrt(np.sum(self.X * self.X, axis=0))
        return self.Z

    def calculate_information_utility(self):
        """计算信息效用值 D"""
        if self.Z is None:  # 如果 Z 未计算，先标准化
            self.standardize()
        n, m = self.Z.shape  # 获取矩阵行列数
        D = np.zeros(m)  # 初始化信息效用值数组
        for i in range(m):
            x = self.Z[:, i]  # 获取第 i 列
            p = x / np.sum(x)  # 归一化得到概率分布
            e = -np.sum(p * self.mylog(p)) / np.log(n)  # 计算熵
            D[i] = 1 - e  # 计算信息效用值
        return D

    def calculate_weights(self):
        """计算权重 W"""
        D = self.calculate_information_utility()
        self.W = D / np.sum(D)  # 归一化得到权重
        return self.W

    def compute(self):
        """执行整个熵权法计算，返回权重 W"""
        self.standardize()
        self.calculate_weights()
        return self.W


# 示例用法
if __name__ == "__main__":
    # 定义指标矩阵 X
    X = [[9, 0, 0, 0], [8, 3, 0.9, 0.5], [6, 7, 0.2, 1]]

    # 创建类实例
    ewm = EntropyWeightMethod(X)

    W = ewm.compute()
    print("权重 W = ")
    print(W)
