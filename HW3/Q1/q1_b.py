import numpy as np
import matplotlib.pyplot as plt
import math


sonar_out = np.loadtxt('data.txt', delimiter=',')
sonar_out.shape

xs = []
ys = []

for i in range(90):
    theta = i * np.pi/180
    xs.append(float(sonar_out[i]) * np.cos(theta))
    ys.append(float(sonar_out[i]) * np.sin(theta))

def pseudo_inverse(X, y):

    W = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)

    return W


W_pseudo = pseudo_inverse(np.array(xs).reshape(-1,1), np.array(ys).reshape(-1,1))

plt.title("b. Pseudo Inverse")
plt.xlabel('X')
plt.ylabel('Y')

plt.scatter(xs, ys)               
plt.plot(xs, np.array(xs).reshape(-1,1).dot(W_pseudo), color='blue')
plt.show()
