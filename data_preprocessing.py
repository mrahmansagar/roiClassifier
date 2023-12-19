# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 16:00:11 2023

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany

"""
import os 
os.sys.path.insert(0, 'E:\\dev\\packages')

from tqdm import tqdm 

import gc

import numpy as np
import matplotlib.pyplot as plt 

import h5py

from PIL import Image
from scipy import ndimage

from tkinter import Tcl
#from random import shuffle

from sklearn.utils import shuffle


from proUtils import utils

root_dir = 'D:\sagar\Data'
scans = os.listdir(root_dir)


path_roi = []
path_notRoi = []

for s in scans:
    scan_path = os.path.join(root_dir, s)
    for r in os.listdir(os.path.join(scan_path, 'roi')):
        path_roi.append((os.path.join(root_dir, s, 'roi', r)))
    try:
        for nr in os.listdir(os.path.join(scan_path, 'not_roi')):
            path_notRoi.append((os.path.join(root_dir, s, 'not_roi', nr)))
    except:
        pass

print('Found ', len(path_roi), ' ROI and ', len(path_notRoi), ' notROI sample')    

# Shuffleing and choosing the sample for test and train 
path_roi = shuffle(path_roi, random_state=3)
path_notRoi = shuffle(path_notRoi, random_state=3)

tmp_roi_path, test_roi_path = path_roi[0:500], path_roi[500:533]
tmp_notRoi_path, test_notRoi_path = path_notRoi[0:500], path_notRoi[500:561]

train_roi_path, val_roi_path = tmp_roi_path[0:425], tmp_roi_path[425:500]
train_notRoi_path, val_notRoi_path = tmp_notRoi_path[0:425], tmp_notRoi_path[425:500]


def norm(v, minVal=None, maxVal=None):
    """
    NORM function takes an array and normalized it between 0-1

    Parameters
    ----------
    v : numpy.ndarray
        Array of N dimension.
    minVal : number 
        Any value that needs to be used as min value for normalization. If no
        value is provided then it uses min value of the given array. The default is None.
    maxVal : number 
        Any value that needs to be used as max value for normalization. If no
        value is provided then it uses max value of the given array. The default is None.

    Returns
    -------
    numpy.ndarray 
        Numpy Array of same dimension as input.

    """
    if minVal == None:
        minVal = v.min()
    
    if maxVal == None:
        maxVal = v.max()
      
    maxVal -= minVal
      
    v = ((v - minVal)/maxVal)
    
    return v


def create_formatted_data(dataPath, xdim=300, ydim=300, zdim=300, resize_factor=(0.5, 0.5, 0.5)):
    
    print('Loading ', len(dataPath), ' Samples.... ')
    
    formattedData = np.zeros(shape=(len(dataPath), int(xdim*resize_factor[0]), int(ydim*resize_factor[1]), int(zdim*resize_factor[2]), 1), dtype=np.float32)

    for i, afile in enumerate(tqdm(dataPath)):
        raw_vol = []
        #sorting the slices according to their names like in windows 
        slices = Tcl().call('lsort', '-dict', os.listdir(afile))
        for aSlice in slices:
            img = Image.open(os.path.join(afile, aSlice))
            imgarray = np.array(img)
            raw_vol.append(imgarray)

        raw_vol = np.asarray(raw_vol)
        raw_vol = np.nan_to_num(raw_vol)
        # raw_vol = np.clip(raw_vol, 0.0005, 0.003)
        raw_vol = ndimage.zoom(raw_vol, resize_factor, order=1)
        # Normalize the data : 0-1
        vol = norm(raw_vol)
        vol = utils.norm8bit(raw_vol)
        # th_vol = vol < 55
        formattedData[i, :, :, :, 0] = vol #th_vol
    
    print('Loaded ', len(dataPath), ' Samples with shape ', formattedData.shape, '\n')
    return formattedData




train_roi = create_formatted_data(train_roi_path)
val_roi = create_formatted_data(val_roi_path)
test_roi = create_formatted_data(test_roi_path)


train_notRoi = create_formatted_data(train_notRoi_path)
val_notRoi = create_formatted_data(val_notRoi_path)
test_notRoi = create_formatted_data(test_notRoi_path)


# Creating the label 
train_roi_label = []
for i in range(len(train_roi_path)):
    train_roi_label.append([1, 0])

train_roi_label = np.array(train_roi_label)


# Creating the label 
val_roi_label = []
for i in range(len(val_roi_path)):
    val_roi_label.append([1, 0])

val_roi_label = np.array(val_roi_label)

# Creating the label 
test_roi_label = []
for i in range(len(test_roi_path)):
    test_roi_label.append([1, 0])

test_roi_label = np.array(test_roi_label)


# Creating the label 
train_notRoi_label = []
for i in range(len(train_notRoi_path)):
    train_notRoi_label.append([0, 1])

train_notRoi_label = np.array(train_notRoi_label)

# Creating the label 
val_notRoi_label = []
for i in range(len(val_notRoi_path)):
    val_notRoi_label.append([0, 1])

val_notRoi_label = np.array(val_notRoi_label)


# Creating the label 
test_notRoi_label = []
for i in range(len(test_notRoi_path)):
    test_notRoi_label.append([0, 1])

test_notRoi_label = np.array(test_notRoi_label)

hf = h5py.File('D:\\sagar\\roiClassifier\\trainData\\TrainValTest_Data_150_150_150_Clip_Norm.hdf5', 'w')
# hf = h5py.File('D:\\sagar\\roiClassifier\\trainData\\TrainValTest_Data_150_150_150_Clip_Norm8bit_th55_bool.hdf5', 'w')
hf.create_dataset('roi', data=train_roi, compression='gzip')
hf.create_dataset('notRoi', data=train_notRoi, compression='gzip')
hf.create_dataset('val_roi', data=val_roi, compression='gzip')
hf.create_dataset('val_notRoi', data=val_notRoi, compression='gzip')
hf.create_dataset('test_roi', data=test_roi, compression='gzip')
hf.create_dataset('test_notRoi', data=test_notRoi, compression='gzip')
hf.close()

hf = h5py.File('D:\\sagar\\roiClassifier\\trainData\\TrainValTest_Label_150_150_150_Clip_Norm.hdf5', 'w')
# hf = h5py.File('D:\\sagar\\roiClassifier\\trainData\\TrainValTest_Label_150_150_150_Clip_Norm8bit_th55_bool.hdf5', 'w')
hf.create_dataset('roi', data=train_roi_label, compression='gzip')
hf.create_dataset('notRoi', data=train_notRoi_label, compression='gzip')
hf.create_dataset('val_roi', data=val_roi_label, compression='gzip')
hf.create_dataset('val_notRoi', data=val_notRoi_label, compression='gzip')
hf.create_dataset('test_roi', data=test_roi_label, compression='gzip')
hf.create_dataset('test_notRoi', data=test_notRoi_label, compression='gzip')
hf.close()


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
                                rotateFormattedVol(train_roi, 45), flipFormattedVol(rotateFormattedVol(train_roi, 45)), 
                                rotateFormattedVol(train_roi, 90), flipFormattedVol(rotateFormattedVol(train_roi, 90)), 
                                rotateFormattedVol(train_roi, 180), flipFormattedVol(rotateFormattedVol(train_roi, 180))),
                                axis=0)

train_label_ROI = np.concatenate((train_roi_label, train_roi_label,
                                  train_roi_label, train_roi_label,
                                  train_roi_label, train_roi_label,
                                  train_roi_label, train_roi_label), axis=0)

#del train_roi
#del train_roi_label
gc.collect()

trainDatanotROI = np.concatenate( (train_notRoi, flipFormattedVol(train_notRoi), 
                                rotateFormattedVol(train_notRoi, 45), flipFormattedVol(rotateFormattedVol(train_notRoi, 45)), 
                                rotateFormattedVol(train_notRoi, 90), flipFormattedVol(rotateFormattedVol(train_notRoi, 90)), 
                                rotateFormattedVol(train_notRoi, 180), flipFormattedVol(rotateFormattedVol(train_notRoi, 180))),
                                axis=0)

train_label_notROI = np.concatenate((train_notRoi_label, train_notRoi_label,
                                  train_notRoi_label, train_notRoi_label,
                                  train_notRoi_label, train_notRoi_label,
                                  train_notRoi_label, train_notRoi_label), axis=0)

hf = h5py.File('E:\\sagar\\Data\\TrainDataAug.hdf5', 'w')
hf.create_dataset('roi', data=trainDataROI, compression='gzip')
hf.create_dataset('notRoi', data=trainDatanotROI, compression='gzip')
hf.close()

hf = h5py.File('E:\\sagar\\Data\\TrainLabelAug.hdf5', 'w')
hf.create_dataset('roi', data=train_label_ROI, compression='gzip')
hf.create_dataset('notRoi', data=train_label_notROI, compression='gzip')
hf.close()

gc.collect()