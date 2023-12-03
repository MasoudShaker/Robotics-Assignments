from controller import Robot, Motor
import math
import matplotlib.pyplot as plt

TIME_STEP = 64
MAX_SPEED = 4

robot = Robot()
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))

gps = robot.getDevice('gps')
gps.enable(TIME_STEP)

compass = robot.getDevice('compass')
compass.enable(TIME_STEP)

left_motor_velocity = 0.9 * MAX_SPEED
right_motor_velocity = MAX_SPEED

leftMotor.setVelocity(left_motor_velocity)
rightMotor.setVelocity(right_motor_velocity)
        
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
    
    if (c > 370 ):
        break
    c += 1
    t += TIME_STEP
    print(t)
    
    if t >= 8000 :
        left_motor_velocity = 0.8 * MAX_SPEED
        right_motor_velocity = 0.1 * MAX_SPEED
        
    if t >= 9200:
        left_motor_velocity = MAX_SPEED
        right_motor_velocity = 0.5 * MAX_SPEED
        
    if t >= 14000:
        left_motor_velocity = 0.9 * MAX_SPEED
        right_motor_velocity = MAX_SPEED
        
    if t >= 17000:
        left_motor_velocity = 0.7 * MAX_SPEED
        right_motor_velocity = 0.8 * MAX_SPEED
        
    if t >= 18000:
        left_motor_velocity = 0.9 * MAX_SPEED
        right_motor_velocity = MAX_SPEED
        
    if t >= 21000:
        left_motor_velocity = 0.1 * MAX_SPEED
        right_motor_velocity = 0.8 * MAX_SPEED
        
    leftMotor.setVelocity(left_motor_velocity)
    rightMotor.setVelocity(right_motor_velocity)

# Plot
fig, ax = plt.subplots(2)

ax[0].set(xlabel='X', ylabel='Y')
ax[0].plot(x, y)

ax[1].plot(time, theta)
ax[1].set(xlabel='T', ylabel='θ')
plt.show()

