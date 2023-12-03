from controller import Robot, Motor
import math
import matplotlib.pyplot as plt

TIME_STEP = 64

MAX_SPEED = 6.28

# create the Robot instance.
robot = Robot()

# get a handler to the motors and set target position to infinity (speed control)
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))

# set up the motor speeds 
leftMotor.setVelocity(1)
rightMotor.setVelocity(1)

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
    x.append(gps_value[0])
    y.append(gps_value[1])

    compass_value = compass.getValues()
    time.append(t)
    theta.append(math.atan2(compass_value[1], compass_value[0]))
    
    if (c > 55):
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