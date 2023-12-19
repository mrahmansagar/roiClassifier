# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 16:45:49 2023

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany
"""

# import necessary libraries 
import os 
os.sys.path.insert(0, 'E:\\dev\\packages')
import shutil
from tkinter import Tcl
from tqdm import tqdm 

import numpy as np

from PIL import Image
from scipy import ndimage
from proUtils import utils

data_dir = 'E:\\sagar\\Data\\MD_1264_A9_Z0.0mm_Z6.6mm\\tiles'
tiles = os.listdir(data_dir)

roi_dir = data_dir.replace('\\tiles', '\\TP')
notRoi_dir = data_dir.replace('\\tiles', '\\TN')

from tensorflow.keras.models import load_model

modelName = 'best_model_09271238.hdf5'
modelPath = 'models/'+ modelName

model = load_model(modelPath)

roi = []
not_roi = []

for atile in tqdm(tiles):
    tile_path = os.path.join(data_dir, atile)
    formattedData = np.zeros(shape=(1, 150, 150, 150, 1), dtype='bool')
    raw_vol = []
    #sorting the slices according to their names like in windows 
    slices = Tcl().call('lsort', '-dict', os.listdir(tile_path))
    for aSlice in slices:
        img = Image.open(os.path.join(tile_path, aSlice))
        imgarray = np.array(img)
        raw_vol.append(imgarray)

    raw_vol = np.asarray(raw_vol)
    raw_vol = np.nan_to_num(raw_vol)
    raw_vol = np.clip(raw_vol, 0.0005, 0.003)
    raw_vol = ndimage.zoom(raw_vol, (0.5, 0.5, 0.5), order=1)
    # Normalize the data : 0-1
    vol = utils.norm8bit(raw_vol)
    th_vol = vol < 55

    formattedData[0, :, :, :, 0] = th_vol

    predicted = model.predict(formattedData)

    if predicted[0][0] >= 0.5:
        shutil.copytree(tile_path, os.path.join(roi_dir, atile))
        roi.append(atile)

    else:
        shutil.copytree(tile_path, os.path.join(notRoi_dir, atile))
        not_roi.append(atile)

classified = {}
classified['roi'] = roi
classified['not_roi'] = not_roi

import json
jsonString = json.dumps(classified)
jsonFile = open(data_dir.split('tiles')[0] + 'Classified_' + modelName.split('.')[0] + '.json', "w")
jsonFile.write(jsonString)
jsonFile.close()