import numpy as np
import matplotlib.pyplot as plt
import math

sonar_out = np.loadtxt('data.txt',delimiter=',')
sonar_out.shape

xs = []
ys = []

for i in range(90): 
    theta = i * np.pi/180
    xs.append(float(sonar_out[i]) * np.cos(theta))
    ys.append(float(sonar_out[i]) * np.sin(theta))


plt.title("a. Plotting the data")
plt.xlabel('X')
plt.ylabel('Y')

plt.scatter(xs, ys)               
plt.show()
