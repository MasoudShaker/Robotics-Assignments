from controller import Robot, Motor
import math
import numpy as np
import matplotlib.pyplot as plt

TIME_STEP = 64

MAX_SPEED = 6.28

# create the Robot instance.
robot = Robot()

# get a handler to the motors and set target position to infinity (speed control)
wheel0 = robot.getDevice('wheel0_joint')
wheel1 = robot.getDevice('wheel1_joint')
wheel2 = robot.getDevice('wheel2_joint')

wheel0.setPosition(float('inf'))
wheel1.setPosition(float('inf'))
wheel2.setPosition(float('inf'))

# set up the motor speeds 
wheel0.setVelocity(1)
wheel1.setVelocity(1)
wheel2.setVelocity(1)

# GPS
gps = robot.getDevice('gps')
gps.enable(TIME_STEP)

# Compass
compass = robot.getDevice('compass')
compass.enable(TIME_STEP)

c = 1
t = 0

# create x, y, theta, time lists
x = []
y = []
theta = []
time = []

while robot.step(TIME_STEP) != -1:
    gps_value = gps.getValues()
    print(gps_value)
    x.append(gps_value[0])
    y.append(gps_value[1])

    compass_value = compass.getValues()
    time.append(t)
    theta.append(math.atan2(compass_value[1], compass_value[0]))
    
    if (c > 100):
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