import math
import numpy as np
import matplotlib.pyplot as plt


def p_controller(x_goal, y_goal, x_initial=0, y_initial=0, theta_initial=math.pi/2, delta_t=0.01, threshold=0.0001):
    x_errors = []
    y_errors = []

    theta = theta_initial
    x_cur = x_initial
    y_cur = y_initial

    e_x = 0
    e_y = 0

    while math.sqrt(math.pow((x_goal-x_cur), 2) + math.pow((y_goal-y_cur), 2)) > threshold:
        delta_x = x_goal - x_cur
        delta_y = y_goal - y_cur

        ro = math.sqrt(math.pow(delta_x, 2) + math.pow(delta_y, 2))

        alpha = -theta + math.atan2(delta_y, delta_x)
        while(alpha > math.pi):
          alpha = alpha - 2 * math.pi
        while(alpha < -math.pi):
          alpha = alpha + 2 * math.pi

        beta = -theta - alpha
        while(beta > math.pi):
          beta = beta - 2 * math.pi
        while(beta < -math.pi):
          beta = beta + 2 * math.pi

        k = (3, 8, -1.5)  # (k_ro, k_alpha, k_beta)

        v = k[0] * ro
        omega = k[1] * alpha + k[2] * beta

        x_dot, y_dot, theta_dot = v * \
            math.cos(theta), v * math.sin(theta), omega

        x_cur, y_cur, theta = x_cur + x_dot*delta_t, y_cur + \
            y_dot*delta_t, theta + omega*delta_t

        e_x += x_goal - x_cur
        e_y = + y_goal - y_cur

        x_errors.append(e_x)
        y_errors.append(e_y)

    return x_errors, y_errors


plt.figure(figsize=(7, 7), dpi=100)

xx, yy = p_controller(0, 10/1000)
plt.plot(xx, yy, color='black')
plt.plot([xx[0], xx[len(xx)-1]], [yy[0], yy[len(yy)-1]], 'orange')

xx, yy = p_controller(0, -10/1000)
plt.plot(xx, yy, color='black')
plt.plot([xx[0], xx[len(xx)-1]], [yy[0], yy[len(yy)-1]], 'orange')

xx, yy = p_controller(10/1000, 0)
plt.plot(xx, yy, color='black')
plt.plot([xx[0], xx[len(xx)-1]], [yy[0], yy[len(yy)-1]], 'orange')

xx, yy = p_controller(-10/1000, 0)
plt.plot(xx, yy, color='black')
plt.plot([xx[0], xx[len(xx)-1]], [yy[0], yy[len(yy)-1]], 'orange')

xx, yy = p_controller(10/1000, 10/1000)
plt.plot(xx, yy, color='black')
plt.plot([xx[0], xx[len(xx)-1]], [yy[0], yy[len(yy)-1]], 'orange')

xx, yy = p_controller(-10/1000, 10/1000)
plt.plot(xx, yy, color='black')
plt.plot([xx[0], xx[len(xx)-1]], [yy[0], yy[len(yy)-1]], 'orange')

xx, yy = p_controller(10/1000, -10/1000)
plt.plot(xx, yy, color='black')
plt.plot([xx[0], xx[len(xx)-1]], [yy[0], yy[len(yy)-1]], 'orange')

xx, yy = p_controller(-10/1000, -10/1000)
plt.plot(xx, yy, color='black')
plt.plot([xx[0], xx[len(xx)-1]], [yy[0], yy[len(yy)-1]], 'orange')
