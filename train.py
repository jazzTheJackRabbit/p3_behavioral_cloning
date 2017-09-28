import pandas as pd
import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Dropout, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
import os
import csv

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                for view in range(0,3):
                    for reverse in range(0,2):
                        name = './data/IMG/'+batch_sample[view].split('/')[-1]
                        center_image = cv2.imread(name)
                        center_angle = float(batch_sample[3])
                        
                        if reverse:
                            center_image = np.fliplr(center_image)
                            center_angle = -center_angle
                        
                        images.append(center_image)
                        angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 160, 320  # Trimmed image format
input_shape = (row, col, ch)

print("Training...")
model = Sequential()

# Preprocess
model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=input_shape, output_shape=input_shape))
model.add(Cropping2D(cropping=((70,25), (0,0))))

# Convolutions
model.add(Convolution2D(24, (5, 5), strides=(2,2), activation='relu'))
model.add(Convolution2D(36, (5, 5), strides=(2,2), activation='relu'))
model.add(Convolution2D(48, (5, 5), strides=(2,2), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Convolution2D(128, (2, 2), activation='relu'))

# Flatten
model.add(Flatten())    

# Fully Connected
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")

model.fit_generator(train_generator, 
        steps_per_epoch=len(train_samples)/32,
        #samples_per_epoch= len(train_samples), 
        validation_data=validation_generator,
        validation_steps=len(validation_samples),
        epochs=3)

model.save('model/model.h5')
