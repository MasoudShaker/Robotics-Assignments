import numpy as np
import matplotlib.pyplot as plt
import math


sonar_out = np.loadtxt('data.txt', delimiter=',')
sonar_out.shape

xs = []
ys = []

for i in range(360):
    theta = i * np.pi/180
    xs.append(float(sonar_out[i]) * np.cos(theta))
    ys.append(float(sonar_out[i]) * np.sin(theta))


def pseudo_inverse(X, y):

    W = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)

    return W


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


def point_line_distance(point, line):
    p1, p2 = np.array(line[0]), np.array(line[1])
    p3 = np.array(point)

    d = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)

    return d


point_line_distance((6, 7), ((6, 7), (4, 6)))

lines = []
splitted_lines = []
points = []
for i in range(len(xs)):
    points.append((xs[i], ys[i]))

lines.append((0, len(points)-1))
threshold = 10

while len(lines) != 0:
    current_line = lines.pop(0)
    max_dist, furthest_point, idx = 0, None, None
    for i in range(current_line[0], current_line[1]+1):
        distance = point_line_distance(point=points[i],
                                       line=(points[current_line[0]], points[current_line[1]]))
        if max_dist < distance:
            max_dist = distance
            furthest_point = points[i]
            idx = i

    if max_dist > threshold: 
        lines.append((current_line[0], idx))
        lines.append((idx, current_line[1]))
    else:
        splitted_lines.append(current_line)


splitted_lines


plt.scatter(xs, ys)  
for line in splitted_lines:
    x_values = [points[line[0]][0], points[line[1]][0]]
    y_values = [points[line[0]][1], points[line[1]][1]]
    plt.plot(x_values, y_values, color='blue')

plt.title("Split")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

line_without_outliers = []
OUTLIERTHRESHOLD = 200
for line in splitted_lines:

    xL1, yL1, xL2, yL2 = points[line[0]][0], points[line[0]
                                                    ][1], points[line[1]][0], points[line[1]][1]
    Llength = math.sqrt((yL2-yL1)**2 + (xL2-xL1)**2)
    if not(abs(line[0]-line[1]) <= 3 or Llength > OUTLIERTHRESHOLD):
        line_without_outliers.append(line)


def find_sum_distance_point(current_line):
    sum_distance = 0
    for i in range(current_line[0], current_line[1]+1):
        dist = point_line_distance(point=points[i], line=(
            points[current_line[0]], points[current_line[1]]))
        sum_distance = sum_distance + dist
    return sum_distance


find_sum_distance_point(splitted_lines[0])


splittedlines = line_without_outliers.copy()
index = 0
thresholdMerge = 0.2
while index != len(splittedlines)-1:
    lineL = splittedlines[index]
    lineR = splittedlines[index+1]
    xL1, yL1, xL2, yL2 = points[lineL[0]][0], points[lineL[0]
                                                     ][1], points[lineL[1]][0], points[lineL[1]][1]
    xR1, yR1, xR2, yR2 = points[lineR[0]][0], points[lineR[0]
                                                     ][1], points[lineR[1]][0], points[lineR[1]][1]
    lineLeftError = find_sum_distance_point(lineL)
    lineRightError = find_sum_distance_point(lineR)
    mergedLine = (lineL[0], lineR[1])
    mergedLineError = find_sum_distance_point(mergedLine)
    Llength = math.sqrt((yL2-yL1)**2 + (xL2-xL1)**2)
    Rlength = math.sqrt((yR2-yR1)**2 + (xR2-xR1)**2)
    Mlenght = math.sqrt((yR2-yL1)**2 + (xR2-xL1)**2)
    LeftError = lineLeftError / Llength
    RightError = lineRightError / Rlength
    MergedError = mergedLineError / Mlenght

    if LeftError + RightError > MergedError + thresholdMerge:  
        splittedlines[splittedlines.index(lineL)] = (lineL[0], lineR[1])
        splittedlines.remove(lineR)
        index = index - 2

    index = index+1
    if(index < 0):
        index = 0

plt.scatter(xs, ys)  
for line in splitted_lines:
    x_values = [points[line[0]][0], points[line[1]][0]]
    y_values = [points[line[0]][1], points[line[1]][1]]
    plt.plot(x_values, y_values, color='blue')

plt.title("Merge")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
