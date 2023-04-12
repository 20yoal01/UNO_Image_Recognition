import numpy as np
import pandas as pd
import gc
import caer
import os
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from tensorflow.python.keras.callbacks import LearningRateScheduler
import canaro
import tensorflow as tf
import cv2 as cv

IMG_SIZE = (80, 80)
channels = 1
char_path = r'simpsons_recognition/input/simpsons_dataset/simpsons_dataset'

char_dict = {}
for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path, char)))
    
char_dict = caer.sort_dict(char_dict, descending=True)

characters = []
count = 0
for i in char_dict:
    characters.append(i[0])
    count += 1
    if count >= 10:
        break
    
train = caer.preprocess_from_dir(char_path, characters, channels=channels, IMG_SIZE=IMG_SIZE, isShuffle=True)

featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)

featureSet = caer.normalize(featureSet)
labels = to_categorical(labels, len(characters))

x_train, x_val, y_train, y_val = caer.train_val_split(featureSet, labels, val_ratio=.2)

del train
del featureSet
del labels
gc.collect()

BATCH_SIZE = 32
EPOCHS = 10

datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train, batch_size = BATCH_SIZE)

model = canaro.models.createSimpsonsModel(IMG_SIZE=IMG_SIZE, channels=channels, output_dim=len(characters), loss='binary_crossentropy', 
                                          decay=1e-6, learning_rate=0.001, momentum=0.9, nesterov=True)

model.summary()

callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]

training = model.fit(train_gen, steps_per_epoch=len(x_train)//BATCH_SIZE, epochs=EPOCHS, validation_data=(x_val, y_val),
                     validation_steps=len(y_val)//BATCH_SIZE, callbacks=callbacks_list)

model.save('simpsons.model')

