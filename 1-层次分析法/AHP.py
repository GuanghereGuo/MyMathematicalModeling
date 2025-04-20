import numpy as np


class AHP:
    """
    层次分析法（Analytic Hierarchy Process, AHP）类，用于计算判断矩阵的权重和一致性检验。
    """

    def __init__(self, A):
        """
        初始化AHP类，接收判断矩阵A。

        参数:
        A (numpy.ndarray): 判断矩阵，要求为方阵。

        异常:
        ValueError: 如果输入矩阵不是方阵，则抛出异常。
        """
        self.A = A
        self.n = A.shape[0]
        if A.shape[0] != A.shape[1]:
            raise ValueError("输入矩阵必须为方阵。")
        # 随机一致性指标RI表
        self.RI = [
            0,  # n=0，未使用
            0,  # n=1
            0.0001,  # n=2
            0.52,  # n=3
            0.89,  # n=4
            1.12,  # n=5
            1.26,  # n=6
            1.36,  # n=7
            1.41,  # n=8
            1.46,  # n=9
            1.49,  # n=10
            1.52,  # n=11
            1.54,  # n=12
            1.56,  # n=13
            1.58,  # n=14
            1.59,  # n=15
        ]

    def check_consistency(self):
        """
        检查判断矩阵的一致性，计算并打印一致性指标CI和一致性比例CR。

        返回:
        bool: 如果一致性比例CR < 0.10，返回True，表示一致性可接受；否则返回False。

        异常:
        ValueError: 如果矩阵阶数n > 15，RI表未定义，则抛出异常。
        """
        eig_val, _ = np.linalg.eig(self.A)
        max_eig = abs(np.max(eig_val))
        # 处理n=1的情况，避免除以零
        CI = (max_eig - self.n) / (self.n - 1) if self.n > 1 else 0
        if self.n > len(self.RI):
            raise ValueError("RI未定义，n > 15")
        RI = self.RI[self.n]
        CR = CI / RI if RI != 0 else 0
        # print(f"一致性指标CI={CI:.3f}")
        # print(f"一致性比例CR={CR:.3f}")
        # if CR < 0.10:
        #     print("因为CR<0.10，所以该判断矩阵A的一致性可以接受!")
        # else:
        #     print("注意：CR >= 0.10，因此该判断矩阵A需要进行修改!")
        # return CR
        return CR < 0.10  # 返回布尔值，表示一致性是否可接受

    def calculate_weights_arithmetic(self):
        """
        使用算术平均法计算权重。

        返回:
        numpy.ndarray: 权重向量。
        """
        a_sum_c = np.sum(self.A, axis=0)  # 按列求和
        Stand_A = self.A / a_sum_c  # 归一化矩阵
        a_sum_r = np.sum(Stand_A, axis=1)  # 按行求和
        weights = a_sum_r / self.n  # 计算权重
        return weights

    def calculate_weights_geometric(self):
        """
        使用几何平均法计算权重。

        返回:
        numpy.ndarray: 权重向量。
        """
        prod_A = np.prod(self.A, axis=1)  # 每行元素相乘
        prod_n_A = np.power(prod_A, 1 / self.n)  # 开n次方
        weights = prod_n_A / np.sum(prod_n_A)  # 归一化
        return weights

    def calculate_weights_eigenvalue(self):
        """
        使用特征值法计算权重。

        返回:
        numpy.ndarray: 权重向量。
        """
        eig_values, eig_vectors = np.linalg.eig(self.A)
        max_index = np.argmax(eig_values)  # 最大特征值索引
        max_vector = eig_vectors[:, max_index]  # 对应特征向量
        weights = max_vector / np.sum(max_vector)  # 归一化
        return weights


# 定义判断矩阵
A = np.array(
    [[1, 2, 3, 5], [1 / 2, 1, 1 / 2, 2], [1 / 3, 2, 1, 2], [1 / 5, 1 / 2, 1 / 2, 1]]
)

# 创建AHP实例
ahp = AHP(A)

# 检查一致性
if ahp.check_consistency():
    print("一致性可接受")
else:
    print("一致性不可接受，请修改判断矩阵")
    exit(0)

# 计算各种权重
weights_arithmetic = ahp.calculate_weights_arithmetic()
weights_geometric = ahp.calculate_weights_geometric()
weights_eigenvalue = ahp.calculate_weights_eigenvalue()

# 打印结果
print("算术平均法权重:", weights_arithmetic)
print("几何平均法权重:", weights_geometric)
print("特征值法权重:", weights_eigenvalue)
