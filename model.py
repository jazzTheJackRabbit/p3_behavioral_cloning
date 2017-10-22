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
from sklearn.model_selection import train_test_split

# Read the log from the simulator
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


# Split the samples into training and validation samples
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    """
    Generator function to generate data for training rather than storing the training data in memory.
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                for view in range(0,3):
                    # For each camera view - Center/Left/Right
                    for reverse in range(0,2):
                        name = './data/IMG/'+batch_sample[view].split('/')[-1]
                        center_image = cv2.imread(name)
                        center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                        center_angle = float(batch_sample[3])

                        # Flip the image horizontally to increase the dataset and help generalize/avoid over-fitting.
                        if reverse:
                            center_image = np.fliplr(center_image)
                            center_angle = -center_angle
                        
                        images.append(center_image)
                        angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


BATCH_SIZE = 32

# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

# Trimmed image format
ch, row, col = 3, 160, 320
input_shape = (row, col, ch)

# Model creation
print("Training...")
model = Sequential()

# Pre-processing
# Lambda layer for normalization
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape, output_shape=input_shape))

# Cropping layer to select only the most relevant region of the frame.
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# Convolution layers with increasing depth sizes to learn increasingly higher level features starting from low-level
# features like lines and curves to shape of the roads and regions to drive within.
model.add(Convolution2D(24, (5, 5), strides=(2,2), activation='relu'))
model.add(Convolution2D(36, (5, 5), strides=(2,2), activation='relu'))
model.add(Convolution2D(48, (5, 5), strides=(2,2), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Convolution2D(128, (2, 2), activation='relu'))

# Flatten Layers
model.add(Flatten())    

# Fully Connected Layers
model.add(Dense(100))

# Drop out layer to help generalize and avoid over-fitting
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")

model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_samples)/BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=len(validation_samples),
    epochs=3
)

model.save('model/model.h5')
