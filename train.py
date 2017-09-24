import pandas as pd
import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten
import numpy as np

def read_images(data):
    images = []

    for image_path in data['center']:
        image = cv2.imread(image_path)
        images.append(image)
        
    return images

def read_steering_measurements(data):
    measurements = []
    
    for measurement in data['steering']:
        measurements.append(measurement)
        
    return measurements

data = pd.read_csv('data/driving_log.csv')
data.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

X = np.array(read_images(data))
y = np.array(read_steering_measurements(data))

model = Sequential()
model.add(Flatten(input_shape = X[0].shape))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X, y, validation_split=0.2, shuffle=True)

model.save('model.h5')