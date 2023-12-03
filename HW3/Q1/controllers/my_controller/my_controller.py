from controller import Robot
import numpy as np
import math

def get_bearing_in_degrees(north):
  rad = math.atan2(north[0], north[2])
  bearing = (rad - 1.5708) / 3.14 * 180.0
  if (bearing < 0.0):
    bearing = bearing + 360.0
    
  return bearing

robot = Robot()

timestep = int(robot.getBasicTimeStep())



left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))        
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

ds0 = robot.getDevice('ds0')
ds1 = robot.getDevice('ds1')
ds2 = robot.getDevice('ds2')
ds3 = robot.getDevice('ds3')
ds4 = robot.getDevice('ds4')
ds5 = robot.getDevice('ds5')
ds6 = robot.getDevice('ds6')
ds7 = robot.getDevice('ds7')
ds8 = robot.getDevice('ds8')
ds9 = robot.getDevice('ds9')
ds0.enable(timestep)
ds1.enable(timestep)
ds2.enable(timestep)
ds3.enable(timestep)
ds4.enable(timestep)
ds5.enable(timestep)
ds6.enable(timestep)
ds7.enable(timestep)
ds8.enable(timestep)
ds9.enable(timestep)

compass = robot.getDevice('compass')
compass.enable(timestep)

values = []
idx = 0
while robot.step(timestep) != -1:
    
    one_round = []
    
    heading = get_bearing_in_degrees(compass.getValues())
    heading = 360 - heading
    
    if abs(heading - idx) <= 0.1:
        one_round.append(ds0.getValue())
        one_round.append(ds1.getValue())
        one_round.append(ds2.getValue())
        one_round.append(ds3.getValue())
        one_round.append(ds4.getValue())
        one_round.append(ds5.getValue())
        one_round.append(ds6.getValue())
        one_round.append(ds7.getValue())
        one_round.append(ds8.getValue())
        one_round.append(ds9.getValue())
        
        values.append(one_round)
        one_round = []
        idx += 1
        
    
    if idx == 36:
        idx += 1
        output = np.array(values).T
        output = output.flatten()
        np.savetxt('data.txt', output, delimiter=',')
        
    left_motor.setVelocity(-0.1)
    right_motor.setVelocity(0.1)
