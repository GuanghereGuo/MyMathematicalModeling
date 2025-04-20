import numpy as np
import sys

A = np.array(
    [[1, 2, 3, 5], [1 / 2, 1, 1 / 2, 2], [1 / 3, 2, 1, 2], [1 / 5, 1 / 2, 1 / 2, 1]]
)

a_sum = np.sum(A, axis=0)  # axis=0表示按列相加
# get a list
# print(a_sum, file=sys.stderr)

n = A.shape[0]  # tuple

# 归一化，二维数组除以一维数组，会自动将一维数组扩展为与二维数组相同的形状，然后进行逐元素的除法运算。
Stand_A = A / a_sum

# 各列相加到同一行
ASumr = np.sum(Stand_A, axis=1)

# 计算权重向量
weights = ASumr / n

print(weights)
