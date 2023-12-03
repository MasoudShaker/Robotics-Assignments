from controller import Robot, Motor
import math
import numpy as np
import matplotlib.pyplot as plt

def inverse_kinematics(linear_velocity, angular_velocity, heading_angle,
    axle_length, wheel_radius):
    
    inertial_frame = [linear_velocity[0],
                              linear_velocity[1],
                              angular_velocity]

    cos_theta = math.cos(heading_angle)
    sin_theta = math.sin(heading_angle)
    rotation_matrix = [[cos_theta, sin_theta, 0],
                       [-sin_theta, cos_theta, 0],
                       [0, 0, 1]]

    robot_frame = np.dot(rotation_matrix, inertial_frame)
    x_dot_r, y_dot_r, theta_dot_r = robot_frame

    phi_dot_1 = (x_dot_r + theta_dot_r * axle_length) / wheel_radius
    phi_dot_2 = (x_dot_r - theta_dot_r * axle_length) / wheel_radius

    return phi_dot_1, phi_dot_2


TIME_STEP = 64

MAX_SPEED = 6.28

# create the Robot instance.
robot = Robot()

# get a handler to the motors and set target position to infinity (speed control)
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))

# GPS
gps = robot.getDevice('gps')
gps.enable(TIME_STEP)

# Compass
compass = robot.getDevice('compass')
compass.enable(TIME_STEP)

phi_dot_1, phi_dot_2 = inverse_kinematics([0, 0], 1, 0, 52, 20.5)

# # set up the motor speeds
leftMotor.setVelocity(phi_dot_1)
rightMotor.setVelocity(phi_dot_2)
        
c = 1
t = 0

# create x, y, theta, time lists
x = []
y = []
theta = []
time = []

while robot.step(TIME_STEP) != -1:
    gps_value = gps.getValues()
    x.append(gps_value[0])
    y.append(gps_value[1])

    compass_value = compass.getValues()
    time.append(t)
    theta.append(math.atan2(compass_value[1], compass_value[0]))
    
    if (c > 50):
        break
    c += 1
    t += TIME_STEP

# Plot
fig, ax = plt.subplots(2)

ax[0].set(xlabel='X', ylabel='Y')
ax[0].plot(x, y)

ax[1].plot(time, theta)
ax[1].set(xlabel='T', ylabel='θ')
plt.show()