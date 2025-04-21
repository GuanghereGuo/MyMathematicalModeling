import numpy as np


class GreyRelationalAnalysis:
    def __init__(self, n, m, kind, A, bestx_list=None, lowx_list=None, highx_list=None):
        """
        初始化GreyRelationalAnalysis类

        参数:
            n (int): 参评数目
            m (int): 指标数目
            kind (list): 指标类型列表，1:极大型, 2:极小型, 3:中间型, 4:区间型
            A (np.ndarray): 原始数据矩阵，形状为 (n, m)
            bestx_list (list, optional): 每个类型3指标的最优值列表，非类型3指标用None占位
            lowx_list (list, optional): 每个类型4指标的区间下限列表，非类型4指标用None占位
            highx_list (list, optional): 每个类型4指标的区间上限列表，非类型4指标用None占位
        """
        self.n = n
        self.m = m
        self.kind = kind
        self.A = A
        self.bestx_list = bestx_list if bestx_list is not None else [None] * m
        self.lowx_list = lowx_list if lowx_list is not None else [None] * m
        self.highx_list = highx_list if highx_list is not None else [None] * m
        self.X = None  # 转换后的极大型指标矩阵
        self.Z = None  # 预处理后的矩阵
        self.Y = None  # 母序列
        self.weight = None  # 权重向量
        self.score = None  # 未归一化的得分
        self.stand_S = None  # 归一化后的得分
        self.sorted_S = None  # 排序后的得分
        self.index = None  # 排序后的索引

    def transform_to_max(self):
        """将所有指标转换为极大型指标"""
        self.X = np.zeros((self.n, self.m))
        for i in range(self.m):
            if self.kind[i] == "1":  # 极大型
                self.X[:, i] = self.A[:, i]
            elif self.kind[i] == "2":  # 极小型
                maxA = max(self.A[:, i])
                self.X[:, i] = self.minTomax(maxA, self.A[:, i])
            elif self.kind[i] == "3":  # 中间型
                if self.bestx_list[i] is None:
                    raise ValueError(f"第{i + 1}个指标（中间型）需要提供最优值")
                self.X[:, i] = self.midTomax(self.bestx_list[i], self.A[:, i])
            elif self.kind[i] == "4":  # 区间型
                if self.lowx_list[i] is None or self.highx_list[i] is None:
                    raise ValueError(f"第{i + 1}个指标（区间型）需要提供区间下限和上限")
                self.X[:, i] = self.regTomax(self.lowx_list[i], self.highx_list[i], self.A[:, i])

    def minTomax(self, maxx, x):
        """极小型指标转换为极大型"""
        return maxx - x

    def midTomax(self, bestx, x):
        """中间型指标转换为极大型"""
        h = np.abs(x - bestx)
        M = max(h)
        if M == 0:
            M = 1
        return 1 - h / M

    def regTomax(self, lowx, highx, x):
        """区间型指标转换为极大型"""
        M = max(lowx - min(x), max(x) - highx)
        if M == 0:
            M = 1
        ans = np.zeros_like(x)
        for i in range(len(x)):
            if x[i] < lowx:
                ans[i] = 1 - (lowx - x[i]) / M
            elif x[i] > highx:
                ans[i] = 1 - (x[i] - highx) / M
            else:
                ans[i] = 1
        return ans

    def preprocess(self):
        """对数据进行预处理"""
        Mean = np.mean(self.X, axis=0)
        self.Z = self.X / Mean

    def construct_sequences(self):
        """构造母序列和子序列"""
        self.Y = np.max(self.Z, axis=1)

    def calculate_correlation(self):
        """计算灰色关联系数"""
        absX0_Xi = np.abs(self.Z - np.tile(self.Y.reshape(-1, 1), (1, self.m)))
        a = np.min(absX0_Xi)
        b = np.max(absX0_Xi)
        rho = 0.5  # 分辨系数
        gamma = (a + rho * b) / (absX0_Xi + rho * b)
        return gamma

    def calculate_weight(self, gamma):
        """计算权重"""
        self.weight = np.mean(gamma, axis=0) / np.sum(np.mean(gamma, axis=0))

    def calculate_score(self):
        """计算得分并排序"""
        self.score = np.sum(self.Z * np.tile(self.weight, (self.n, 1)), axis=1)
        self.stand_S = self.score / np.sum(self.score)
        self.sorted_S = np.sort(self.stand_S)[::-1]
        self.index = np.argsort(self.stand_S)[::-1]

    def run(self):
        """运行整个灰色关联分析流程"""
        self.transform_to_max()
        self.preprocess()
        self.construct_sequences()
        gamma = self.calculate_correlation()
        self.calculate_weight(gamma)
        self.calculate_score()
        print('归一化后的得分及其索引（降序）：')
        print(self.sorted_S)
        print(self.index)


def get_user_input():
    n = eval(input("请输入参评数目："))
    m = eval(input("请输入指标数目："))
    kind = input("请输入类型矩阵（1:极大型，2:极小型，3:中间型，4:区间型）：").split()
    A = np.zeros((n, m))
    print("请输入矩阵：")
    for i in range(n):
        A[i] = list(map(float, input().split()))

    bestx_list = [None] * m
    lowx_list = [None] * m
    highx_list = [None] * m
    for i in range(m):
        if kind[i] == "3":
            bestx_list[i] = eval(input(f"请输入第{i + 1}个指标（中间型）的最优值："))
        elif kind[i] == "4":
            lowx_list[i] = eval(input(f"请输入第{i + 1}个指标（区间型）的区间下限："))
            highx_list[i] = eval(input(f"请输入第{i + 1}个指标（区间型）的区间上限："))

    return n, m, kind, A, bestx_list, lowx_list, highx_list


# 使用交互式输入
n, m, kind, A, bestx_list, lowx_list, highx_list = get_user_input()
gra = GreyRelationalAnalysis(n, m, kind, A, bestx_list, lowx_list, highx_list)
gra.run()