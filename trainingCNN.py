# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 16:25:48 2023

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany

First training with CNN
"""

# only run once and the restart the kernel 
from numba import cuda
devices = cuda.get_current_device()
devices.reset()

import time
import os
import gc 
import h5py
import json

import numpy as np


import matplotlib.pyplot as plt 

from scipy import ndimage
from sklearn.utils import shuffle

# Reading the train data 
hf = h5py.File('D:\\sagar\\roiClassifier\\trainData\\TrainValTest_Data_150_150_150_Clip_Norm8bit_th55_bool.hdf5', 'r')
hf.keys
train_roi = np.array(hf.get('roi'))
train_notRoi = np.array(hf.get('notRoi'))
val_roi = np.array(hf.get('val_roi'))
val_notRoi = np.array(hf.get('val_notRoi'))
test_roi = np.array(hf.get('test_roi'))
test_notRoi = np.array(hf.get('test_notRoi'))

# Reading the train data 
hf = h5py.File('D:\\sagar\\roiClassifier\\trainData\\TrainValTest_Label_150_150_150_Clip_Norm8bit_th55_bool.hdf5', 'r')
hf.keys
train_roi_label = np.array(hf.get('roi'))
train_notRoi_label = np.array(hf.get('notRoi'))
val_roi_label = np.array(hf.get('val_roi'))
val_notRoi_label = np.array(hf.get('val_notRoi'))
test_roi_label = np.array(hf.get('test_roi'))
test_notRoi_label = np.array(hf.get('test_notRoi'))


# Data Augmentation : 8 Set Possible
# 1. Original Data 
# 2,3,4,5 Rotate 45, 90, 180 
# 6, 7, 8 Flip 

aVol = train_roi[0, :, :, :, 0]
fig, ax = plt.subplots(2,4, figsize=(16,9))
ax[0,0].imshow(aVol[75, :, :,], cmap='gray')
ax[0,0].set_title('Orig')

rot45 = ndimage.rotate(aVol, 45, reshape=False)
ax[0,1].imshow(rot45[75, :, :], cmap='gray')
ax[0,1].set_title('rot 45')

rot90 = ndimage.rotate(aVol, 90, reshape=False)
ax[0,2].imshow(rot90[75, :, :], cmap='gray')
ax[0,2].set_title('rot 90')

rot180 = ndimage.rotate(aVol, 180, reshape=False)
ax[0,3].imshow(rot180[75, :, :], cmap='gray')
ax[0,3].set_title('rot 180')

###########

flip = np.flip(aVol)
ax[1,0].imshow(flip[75, :, :], cmap='gray')
ax[1,0].set_title('Flip Orig')


flip45 = np.flip(rot45)
ax[1,1].imshow(flip45[75, :, :], cmap='gray')
ax[1,1].set_title('Flip 45')

flip90 = np.flip(rot90)
ax[1,2].imshow(flip90[75, :, :], cmap='gray')
ax[1,2].set_title('Flip 90')

flip180 = np.flip(rot180)
ax[1,3].imshow(flip180[75, :, :], cmap='gray')
ax[1,3].set_title('Flip 180')

plt.show()

plt.close()


# Rotate formatted volume 
def rotateFormattedVol(arr, angle, reshape=False):
    result = np.empty_like(arr)
    for i in range(arr.shape[0]):
        vol = arr[i, :, :, :, 0]
        vol = ndimage.rotate(vol, angle, reshape=reshape)
        vol[ vol < 0] = 0
        vol[ vol > 1] = 1
        result[i, :, :, :, 0] = vol 
        
    return result


# Flip formatted volume 
def flipFormattedVol(arr):
    result = np.empty_like(arr)
    for i in range(arr.shape[0]):
        vol = arr[i, :, :, :, 0]
        vol = np.flip(vol)
        result[i, :, :, :, 0] = vol 
    
    return result


trainDataROI = np.concatenate( (train_roi, flipFormattedVol(train_roi),  
                                rotateFormattedVol(train_roi, 90), flipFormattedVol(rotateFormattedVol(train_roi, 90)), 
                                rotateFormattedVol(train_roi, 180), flipFormattedVol(rotateFormattedVol(train_roi, 180))),
                                axis=0)

train_label_ROI = np.concatenate((train_roi_label, train_roi_label,
                                  train_roi_label, train_roi_label,
                                  train_roi_label, train_roi_label), axis=0)


del train_roi
del train_roi_label


gc.collect()

trainDatanotROI = np.concatenate( (train_notRoi, flipFormattedVol(train_notRoi),  
                                rotateFormattedVol(train_notRoi, 90), flipFormattedVol(rotateFormattedVol(train_notRoi, 90)), 
                                rotateFormattedVol(train_notRoi, 180), flipFormattedVol(rotateFormattedVol(train_notRoi, 180))),
                                axis=0)

train_label_notROI = np.concatenate((train_notRoi_label, train_notRoi_label,
                                  train_notRoi_label, train_notRoi_label,
                                  train_notRoi_label, train_notRoi_label), axis=0)


del train_notRoi
del train_notRoi_label
gc.collect()

# Training data preparation 
# trainFeatures = np.concatenate((train_roi, train_notRoi), axis=0)
# valFeatures = np.concatenate((val_roi, val_notRoi), axis=0)

trainFeatures = np.concatenate((trainDataROI, trainDatanotROI), axis=0)
valFeatures = np.concatenate((val_roi, val_notRoi), axis=0)

# del train_roi
# del train_notRoi

del trainDataROI
del trainDatanotROI

del val_roi
del val_notRoi

gc.collect()

# trainLabels = np.concatenate((train_roi_label, train_notRoi_label), axis=0)
# valLabels = np.concatenate((val_roi_label, val_notRoi_label), axis=0)

trainLabels = np.concatenate((train_label_ROI, train_label_notROI), axis=0)
valLabels = np.concatenate((val_roi_label, val_notRoi_label), axis=0)

# del train_roi_label
# del train_notRoi_label

del train_label_ROI
del train_label_notROI

del val_roi_label
del val_notRoi_label

gc.collect()

from sklearn.utils import shuffle

X_train, y_train = shuffle(trainFeatures, trainLabels, random_state=2)
X_val, y_val = shuffle(valFeatures, valLabels, random_state=2)

del trainFeatures
del trainLabels
del valFeatures
del valLabels
gc.collect()

import tensorflow as tf 
print(tf.__version__)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization 
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam

# Building a model 
input_shape=X_train.shape[1:]
model = models.Sequential()
# Conv Layers 
model.add(layers.Conv3D(8, (3, 3, 3), padding='same', activation ='relu', input_shape=input_shape, data_format='channels_last'))
model.add(layers.MaxPooling3D(pool_size=(2,2,2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv3D(16, (3, 3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling3D(pool_size=(2,2,2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling3D(pool_size=(2,2,2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling3D(pool_size=(3,3,3)))
model.add(layers.BatchNormalization())


# FC layer 
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(rate = 0.1))

model.add(layers.Dense(64, activation='relu', kernel_regularizer='l2'))
model.add(layers.Dropout(rate = 0.1))
model.add(layers.Dense(2, activation='softmax', kernel_regularizer='l2'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

model.summary()


from tensorflow.keras.callbacks import CSVLogger 
csvFile = 'logs/train_150_clip_norm8bit_bool_Aug_' + time.strftime('%m%d%H%M') + '.csv'
csv_logger = CSVLogger(csvFile, append=True, separator=';')

def print_summary(s):
    txt_file = csvFile.replace('.csv', '.txt')
    with open(txt_file,'a') as f:
        print(s, file=f)

model.summary(print_fn=print_summary)

# Training the model 
epochs = 50
batch_size = 16
hist = model.fit(X_train, y_train, batch_size=batch_size,
                 epochs=epochs, verbose=1, callbacks=[csv_logger],
                 validation_data=(X_val, y_val))

gc.collect()

modelName = 'models/CNNfirstPass_09261216_last.hdf5' # + time.strftime('%m%d%H%M') + '.hdf5'
model.save(modelName)

# Display trainig statistics 
loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

epochsr = range(epochs)

plt.figure()
plt.plot(epochsr, loss, 'bo', label='training loss')
plt.plot(epochsr, val_loss, 'b', label='val. loss')
plt.title('train vs val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.grid('ON')
plt.show()


plt.figure()
plt.plot(epochsr, acc, 'bo', label='training acc')
plt.plot(epochsr, val_acc, 'b', label='val. acc')
plt.title('train vs val acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
plt.grid('ON')
plt.show()

# Model evaluation 
X_test = np.concatenate((test_roi, test_notRoi), axis=0)
y_test = np.concatenate((test_roi_label, test_notRoi_label), axis=0)

scores = model.evaluate(X_test, y_test)
print('Test %s: %.2f%%' % (model.metrics_names[1], scores[1]*100))