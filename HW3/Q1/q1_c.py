import numpy as np
import matplotlib.pyplot as plt
import math

sonar_output = np.loadtxt('data.txt', delimiter=',')
sonar_output.shape

xs = []
ys = []

for i in range(90):
    theta = i * np.pi/180
    xs.append(float(sonar_output[i]) * np.cos(theta))
    ys.append(float(sonar_output[i]) * np.sin(theta))
    

def pseudo_inverse(X, y):

    W = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)

    return W


W_pseudo = pseudo_inverse(np.array(xs).reshape(-1, 1),
                          np.array(ys).reshape(-1, 1))


def compute_MSE(X, y, y_hat):

    m = X.shape[0] 

    MSE = (1 / (2 * m)) * np.sum((y - y_hat)**2)
    return MSE


def ransac(X, y, num_sample, threshold):

    iterations = math.inf
    iterations_done = 0

    max_inlier_count = 0
    best_model = None

    prob_outlier = 0.5
    desired_prob = 0.95

    total_data = np.column_stack((X, y)) 
    data_size = len(total_data)

    iterations_done = 0

    while iterations > iterations_done:

        np.random.shuffle(total_data)

        sample_data = total_data[:num_sample, :]
        X = sample_data[:, :-1]
        y = sample_data[:, -1:]

        estimated_model = pseudo_inverse(X, y)

        y_hat = X.dot(estimated_model)
        MSE = compute_MSE(X, y, y_hat)
        inlier_count = np.count_nonzero(MSE < threshold)

        if inlier_count > max_inlier_count:
            max_inlier_count = inlier_count
            best_model = estimated_model

        prob_outlier = 1 - inlier_count / data_size
        iterations = math.log(1 - desired_prob) / \
            math.log(1 - (1 - prob_outlier) ** num_sample)
        iterations_done += 1

    return best_model


W_ransac = ransac(np.array(xs).reshape(-1, 1),
                  np.array(ys).reshape(-1, 1), num_sample=2, threshold=1000000)


plt.title("c. Ransac")
plt.xlabel('X')
plt.ylabel('Y')

plt.scatter(xs, ys)  
plt.plot(xs, np.array(xs).reshape(-1, 1).dot(W_ransac), color='red')
plt.show()
