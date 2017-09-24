import pandas as pd
import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Dropout, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

import numpy as np
import os

def read_data(data):
    print('Reading data...')
    images = []
    measurements = []

    for row in data.iterrows():
        for camera_path in ['center', 'left', 'right']:
            for reverse in range(0,2):
                image_path = row[1][camera_path] 
                measurement = row[1]['steering']

                image_path = os.path.join(os.path.realpath('./'),'/'.join(image_path.split('/')[-3:]))
                image = cv2.imread(image_path)
                
                if reverse:
                    image = np.fliplr(image)
                    measurement = -measurement

                images.append(image)
                measurements.append(measurement)

    return images, measurements


data = pd.read_csv('data/driving_log.csv')
data.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

data_tuple = read_data(data)
X = np.array(data_tuple[0])
y = np.array(data_tuple[1])

input_shape = X[0].shape

print("Training...")
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))
model.add(Convolution2D(6, (5, 5), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, (5, 5), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())	
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")
model.fit(X,y,validation_split=0.3,epochs=5)

model.save('model/model.h5')
