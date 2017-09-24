import pandas as pd
import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda
import numpy as np
import os

def read_images(data):
    print('Reading images...')
    images = []

    for image_path in data['center']:
        image_path = os.path.join(os.path.realpath('./'),'/'.join(image_path.split('/')[-3:]))
        image = cv2.imread(image_path)
        images.append(image)
        
    return images

def read_steering_measurements(data):
    print('Reading measurements...')
    measurements = []
    
    for measurement in data['steering']:
        measurements.append(measurement)
        
    return measurements

data = pd.read_csv('data/driving_log.csv')
data.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

X = np.array(read_images(data))
y = np.array(read_steering_measurements(data))

input_shape = X[0].shape

print("Training...")
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape))
model.add(Flatten())
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X, y, validation_split=0.3, shuffle=True)

model.save('model/model.h5')
