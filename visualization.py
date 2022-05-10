import numpy as np
import matplotlib.pyplot as plt
import os.path

# bar chart
size = 5
x = np.arange(size)
a = np.random.random(size)
b = np.random.random(size)
c = np.random.random(size)

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.bar(x, a,  width=width, label='a')
plt.bar(x + width, b, width=width, label='b')
plt.bar(x + 2 * width, c, width=width, label='c')
plt.legend()

filename = "test.png"
path = os.getcwd() + "\Co-Design-for-DNN-GCN\img\\"

plt.savefig(path+filename)
plt.show()

#
x_axis_data = [i for i in range(10)]
y_axis_data1 = [12,17,15,12,16,14,15,13,18,19]
y_axis_data2 = [1,4,2,6,4,2,1,6,4,2]

plt.plot(x_axis_data, y_axis_data1)
plt.plot(x_axis_data, y_axis_data2)
plt.show()
