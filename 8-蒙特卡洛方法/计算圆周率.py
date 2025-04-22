import numpy as np
import matplotlib.pyplot as plt

p = 100000 # 采样点数
r = 1
x0, y0 = 0, 0
n = 0

plt.figure()
plt.title('Monte Carlo Method to Estimate Pi')
plt.xlabel('x')
plt.ylabel('y')

# plt.xlim(-r, r)
# plt.ylim(-r, r)

for i in range(p):
    x = np.random.uniform(-r, r)
    y = np.random.uniform(-r, r)
    if x ** 2 + y ** 2 <= r ** 2:
        n += 1
        #plt.scatter(x, y, s=1, c='b')
        plt.plot(x, y, '.', color='b')
    else:
        # plt.scatter(x, y, s=1, c='r')
        plt.plot(x, y, '.', color='r')

plt.axis('equal')
plt.show()

pi = 4 * n / p
print('Estimated Pi:', pi)