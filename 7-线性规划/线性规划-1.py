import numpy as np
import scipy.optimize as opt

c = np.array([-20, -30, -45])  # 目标函数系数

A_ub = np.array([[4, 8, 15], [1, 1, 1]])

b_ub = np.array([100, 20])  # 不等式约束的右侧值
bounds = [(0, None), (0, None), (0, None)]  # 变量的边界

result = opt.linprog(c, A_ub, b_ub, bounds=bounds)
# print(result)
print(result.x)
print(-result.fun)
