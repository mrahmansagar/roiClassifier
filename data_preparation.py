# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:49:36 2023

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany

Seleting the not roi cubes for each scan.
"""

import os 
os.sys.path.insert(0, 'E:\\dev\\packages')

from PIL import Image
import numpy as np
from tqdm import tqdm 
import json 

import gc 
import random

from proUtils import utils


# Path to the slices 
data_dir = 'D:\sagar\Data'

# Get the directories with roi and tiles.json from D
scans = []
for p in os.listdir(data_dir):
    tile_path = os.path.join(data_dir, p, 'tiles.json')
    if os.path.exists(tile_path):
        scans.append(os.path.join(data_dir, p))
        
completed = ['D:\\sagar\\Data\\MD_1264_A10_Z6.6mm',
             'D:\\sagar\\Data\\MD_1264_A11_Z3.3mm_corr_phrt ', 
             'D:\\sagar\\Data\\MD_1264_A12_Z3.3mm_corr_phrt', 
             'D:\\sagar\\Data\\MD_1264_A13_1_Z3.3mm_corr_phrt',
             'D:\\sagar\\Data\\MD_1264_A16_Z3.3mm_corr_phrt',
             'D:\\sagar\\Data\\MD_1264_A18',
             'D:\\sagar\\Data\\MD_1264_A2_1_Z3.3mm',
             'D:\\sagar\\Data\\MD_1264_A11_Z3.3mm_corr_phrt',
             'D:\\sagar\\Data\\MD_1264_A9_Z0.0mm_Z3.3mm',
             'D:\\sagar\\Data\\MD_1264_B5_1_Z3.3mm',
             'D:\\sagar\\Data\\MD_1264_B1_1_Z3.3mm_corr_phrt',
             
             ]

scans_now = list(set(scans) - set(completed))

# Loop through the directories by replacing path name to load the Slices from Drive Sagar
count = 1
for s in scans_now:
    print('procrssing ', count, '/', len(scans_now), ':  file', s, '\n')
    count += 1

    scan_dir = os.path.join('H:', s.split('Data')[1], 'slices')
    slices = os.listdir(scan_dir)
    
    #    first read one slice to get the shape 
    im = Image.open(os.path.join(scan_dir, slices[0]))
    im = np.array(im)
    
    
    #    initialize the volume and load the whole volume 
    vol = np.empty(shape=(1700, im.shape[0], im.shape[1]), dtype=np.float32)
    
    for i, fname in enumerate(tqdm(slices)):
        img = Image.open(os.path.join(scan_dir, fname))
        imgarray = np.array(img)
        vol[i, :, :] = imgarray
        
    #    seperate roi volume form tiles 
    tiles_path = os.path.join(s, 'tiles.json')
    f = open(tiles_path)
    data = json.load(f)
    all_tiles = data['roi']
    f.close()
    rois = os.listdir(os.path.join(s, 'roi'))
    tiles_wo_roi = list(set(all_tiles) - set(rois))
    
    #    Choose tiles randomly : Number of choosen tiles should be 1.3 times the roi number. (rounded int)
    selected_tiles = random.sample(tiles_wo_roi, round(len(rois)*1.3))
    
    #    Save the tiles to drive D in the same scan folder.
    for t in selected_tiles:
        cords = t.split('x')
        zcord, ycord, xcord = cords[0], cords[1], cords[2]
        tile_vol = vol[int(zcord.split('-')[0]):int(zcord.split('-')[1]), int(ycord.split('-')[0]):int(ycord.split('-')[1]), int(xcord.split('-')[0]):int(xcord.split('-')[1]),]
        tiles_dir = os.path.join(s, 'not_roi', t)
        utils.save_vol_as_slices(tile_vol, tiles_dir)
    # Free up some memory
    
    del vol
    gc.collect()