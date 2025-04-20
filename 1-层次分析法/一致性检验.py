import sys

import numpy as np

A = np.array(
    [[1, 2, 3, 5], [1 / 2, 1, 1 / 2, 2], [1 / 3, 2, 1, 2], [1 / 5, 1 / 2, 1 / 2, 1]]
)
n = A.shape[0]  # 0 -> row, 1 -> col

eig_val, eig_vec = np.linalg.eig(A)  # eigenvalue

# print(f"eig_val = {eig_val}", file=sys.stderr)
# print(f"eig_vec = {eig_vec}", file=sys.stderr)

max_eig = abs(np.max(eig_val))  # max Modulus

print(max_eig, file=sys.stderr)

CI = (max_eig - n) / (n - 1)
RI = [
    0,  # indexed from 1
    0,  # n = 1
    0.0001,  # 这里n = 2时，一定是一致矩阵，所以CI = 0
    0.52,
    0.89,
    1.12,
    1.26,
    1.36,
    1.41,
    1.46,
    1.49,
    1.52,
    1.54,
    1.56,
    1.58,
    1.59,
]

CR = CI / RI[n]

print(f"一致性指标CI={CI:.3f}")
print(f"一致性比例CR={CR:.3f}")
print(type(CI), file=sys.stderr)

if CR < 0.10:
    print("因为CR<0.10，所以该判断矩阵A的一致性可以接受!")
else:
    print("注意：CR >= 0.10，因此该判断矩阵A需要进行修改!")
