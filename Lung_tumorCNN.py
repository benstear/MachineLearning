#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 11:35:06 2018

A Convolutional Neural Network for classifying lung tumors as either benign or malignant

@author: benstear
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam, SGD
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
#from keras.callbacks import EarlyStopping, TensorBoard


# Augment and preprocessing w ImageDataGenerator
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)# do not alter test images, only rescale pxl vals
 
train_batchsize = 200 # trainbatch= go as big as you can
val_batchsize = 75
image_size = 256
 
train_generator = train_datagen.flow_from_directory(
        '/Users/dawnstear/desktop/trainCNN/',
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(
        '/Users/dawnstear/desktop/testCNN/',
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='binary',
        shuffle=False)

nb_train_samples = train_generator.samples
nb_validation_samples = validation_generator.samples

# Converting the labels to one-hot encoded matrix
train_labels = np_utils.to_categorical(train_generator.classes)
validation_labels = np_utils.to_categorical(validation_generator.classes)

input_shape = (image_size, image_size, 3)

# Create Model
model = Sequential()

model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=input_shape, name='Conv2D_Layer1'))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# Compile the model
model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.005), metrics=['acc'])

# Train the model
history = model.fit_generator(
      train_generator,
      steps_per_epoch=nb_train_samples/train_generator.batch_size ,
      epochs=4,
      validation_data=validation_generator,
      validation_steps=nb_validation_samples/validation_generator.batch_size,
      verbose=1, shuffle=True)
 
# Save the model
model.save('myModel.h5')

# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)

score = model.evaluate_generator(validation_generator, nb_validation_samples/val_batchsize)
scores = model.predict_generator(validation_generator, nb_validation_samples/val_batchsize)
print("Loss: ", score[0], "Accuracy: ", score[1])






