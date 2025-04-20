import numpy as np

# 定义判断矩阵A
A = np.array(
    [[1, 2, 3, 5], [1 / 2, 1, 1 / 2, 2], [1 / 3, 2, 1, 2], [1 / 5, 1 / 2, 1 / 2, 1]]
)

# 获取A的行和列
n, _ = A.shape

# 求出特征值和特征向量
eig_values, eig_vectors = map(abs, np.linalg.eig(A))
# every column of eig_vectors is an eigenvector

# 找出最大特征值的索引
max_index = np.argmax(eig_values)

# 找出对应的特征向量
max_vector = eig_vectors[:, max_index]
# select the max_index column of eig_vectors

# 对特征向量进行归一化处理,得到权重
weights = max_vector / np.sum(max_vector)

# 输出权重
print(weights)
