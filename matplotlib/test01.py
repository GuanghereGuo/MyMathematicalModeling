import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
#x = np.arange(-10, 10, 0.1)
y = np.sin(x)

fig, ax = plt.subplots()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 移动底部轴线到 y=0
ax.spines['bottom'].set_position(('data', 0))

# 移动左侧轴线到 x=0
ax.spines['left'].set_position(('data', 0))

plt.plot(x, y, label='sin(x)', color='blue', linestyle='-', linewidth=2)
plt.title('Sine Function')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid()
plt.show()
