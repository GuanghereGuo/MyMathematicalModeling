import numpy as np


class Topsis:
    def __init__(
        self,
        n,
        m,
        kind,
        A,
        weight=None,
        bestx_list=None,
        lowx_list=None,
        highx_list=None,
    ):
        """
        初始化Topsis类

        参数:
            n (int): 参评数目
            m (int): 指标数目
            kind (list): 指标类型列表，1:极大型, 2:极小型, 3:中间型, 4:区间型
            A (np.ndarray): 原始数据矩阵，形状为 (n, m)
            weight (list, optional): 权重向量，默认None时每个指标权重为1
            bestx_list (list, optional): 每个类型3指标的最优值列表，非类型3指标用None占位
            lowx_list (list, optional): 每个类型4指标的区间下限列表，非类型4指标用None占位
            highx_list (list, optional): 每个类型4指标的区间上限列表，非类型4指标用None占位
        """
        self.n = n
        self.m = m
        self.kind = kind
        self.A = A
        self.weight = weight if weight is not None else [1] * m
        self.bestx_list = bestx_list if bestx_list is not None else [None] * m
        self.lowx_list = lowx_list if lowx_list is not None else [None] * m
        self.highx_list = highx_list if highx_list is not None else [None] * m
        self.X = None
        self.X_std = None
        self.X_weighted = None
        self.d_z = None
        self.d_f = None
        self.s = None
        self.Score = None

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
                    raise ValueError(f"第{i+1}个指标（中间型）需要提供最优值")
                self.X[:, i] = self.midTomax(self.bestx_list[i], self.A[:, i])
            elif self.kind[i] == "4":  # 区间型
                if self.lowx_list[i] is None or self.highx_list[i] is None:
                    raise ValueError(f"第{i+1}个指标（区间型）需要提供区间下限和上限")
                self.X[:, i] = self.regTomax(
                    self.lowx_list[i], self.highx_list[i], self.A[:, i]
                )

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

    def standardize(self):
        """对数据进行标准化处理"""
        self.X_std = np.zeros_like(self.X)
        for j in range(self.m):
            self.X_std[:, j] = self.X[:, j] / np.sqrt(np.sum(self.X[:, j] ** 2))

    def weight_matrix(self):
        """对标准化后的数据进行加权"""
        self.X_weighted = self.X_std * self.weight

    def calculate_distance(self):
        """计算到最优解和最劣解的距离"""
        x_max = np.max(self.X_weighted, axis=0)
        x_min = np.min(self.X_weighted, axis=0)
        self.d_z = np.sqrt(np.sum((self.X_weighted - x_max) ** 2, axis=1))
        self.d_f = np.sqrt(np.sum((self.X_weighted - x_min) ** 2, axis=1))

    def calculate_score(self):
        """计算得分和百分制得分"""
        self.s = self.d_f / (self.d_z + self.d_f)
        self.Score = 100 * self.s / np.sum(self.s)

    def run(self):
        """运行整个TOPSIS流程"""
        self.transform_to_max()
        self.standardize()
        self.weight_matrix()
        self.calculate_distance()
        self.calculate_score()
        for i in range(self.n):
            print(f"第{i+1}个标准化后百分制得分为：{self.Score[i]}")


def get_user_input():
    n = eval(input("请输入参评数目："))
    m = eval(input("请输入指标数目："))
    kind = input("请输入类型矩阵（1:极大型，2:极小型，3:中间型，4:区间型）：").split()
    A = np.zeros((n, m))
    print("请输入矩阵：")
    for i in range(n):
        A[i] = list(map(float, input().split()))
    flag = input("是否需要给指标加权？(y/N):") == "y"
    weight = [1] * m
    if flag:
        weight = list(map(float, input("请输入权重向量：").split()))

    bestx_list = [None] * m
    lowx_list = [None] * m
    highx_list = [None] * m
    for i in range(m):
        if kind[i] == "3":
            bestx_list[i] = eval(input(f"请输入第{i+1}个指标（中间型）的最优值："))
        elif kind[i] == "4":
            lowx_list[i] = eval(input(f"请输入第{i+1}个指标（区间型）的区间下限："))
            highx_list[i] = eval(input(f"请输入第{i+1}个指标（区间型）的区间上限："))

    return n, m, kind, A, weight, bestx_list, lowx_list, highx_list


# 使用交互式输入
n, m, kind, A, weight, bestx_list, lowx_list, highx_list = get_user_input()
topsis = Topsis(n, m, kind, A, weight, bestx_list, lowx_list, highx_list)
topsis.run()
