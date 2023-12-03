from controller import Robot
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from keras.models import load_model

def make_prediction(img_dir):
    model = load_model('E:/University/Term_8/Robotix/Assignments/HW3/ex3/Q2/worlds/CNN_q2.h5')
   
    test_img = image.load_img(img_dir, target_size=(img_size, img_size))
    test_img = test_img.convert(mode='RGB')
    test_img = tf.keras.preprocessing.image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    test_img = tf.keras.applications.inception_v3.preprocess_input(test_img)

    result = model.predict(test_img)

    if result[0][0] > result[0][1] + result[0][2] + result[0][3]:
        return 'circle'
    elif result[0][1] > result[0][0] + result[0][2] + result[0][3]:
        return 'square'
    elif result[0][2] > result[0][0] + result[0][1] + result[0][3]:
        return 'star'
    elif result[0][3] > result[0][0] + result[0][1] + result[0][2]:
        return 'triangle'

img_size = 28
timestep = 64

robot = Robot()
        
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

left_motor.setVelocity(0)
right_motor.setVelocity(0)

camera = robot.getDevice('camera_1')
camera.enable(timestep)

t = 0

while robot.step(timestep) != -1 and t < 1:
    t+=1
    saved_img = camera.saveImage('E:/University/Term_8/Robotix/Assignments/HW3/ex3/Q2/worlds//saved_img.png', 100)
    print(make_prediction('E:/University/Term_8/Robotix/Assignments/HW3/ex3/Q2/worlds//saved_img.png'))