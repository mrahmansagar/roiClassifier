# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 17:37:03 2022

@author: Mmr Sagar
PhD Student | AG Alves 
MPI for Multidisciplinary Sciences, Germany 


Plots of training history 

"""


# import necessary libraries 
import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns



def extract_csvlog(fileName):
  epochsr = []
  loss = []
  val_loss = []
  acc = []
  val_acc = []

  with open(fileName) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=';') 

    for row in readCSV:
        ep = row[0]
        ac = row[1]
        lo = row[2]
        vlac = row[3]
        vllo = row[4]

        epochsr.append(ep)
        loss.append(lo)
        val_loss.append(vllo)
        acc.append(ac)
        val_acc.append(vlac)


  epochsr = (np.array(epochsr[1:])).astype(np.float32)
  loss = (np.array(loss[1:])).astype(np.float32)
  val_loss = (np.array(val_loss[1:])).astype(np.float32)
  acc = (np.array(acc[1:])).astype(np.float32)
  val_acc = (np.array(val_acc[1:])).astype(np.float32)

  return epochsr, loss, val_loss, acc, val_acc


epochsr1, loss1, val_loss1, acc1, val_acc1 = extract_csvlog(fileName='logs/train_150_clip_norm_noAug_09191728.csv')
epochsr2, loss2, val_loss2, acc2, val_acc2 = extract_csvlog(fileName='logs/train_150_clip_norm_noAug_09191739.csv')  
#epochsr3, loss3, val_loss3, acc3, val_acc3 = extract_csvlog(fileName='logs/train_150_clip_norm_Aug_09201142.csv')

sns.set(rc={'figure.figsize':(8,5)})
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
markersize = 3
linewidth = 1

plt.figure()
plt.plot(epochsr1[0:20], loss1[0:20], 'bo-', markersize=markersize, label='train loss', linewidth=linewidth)
plt.plot(epochsr1[0:20], val_loss1[0:20], 'bs-', markersize=markersize, label='val loss', linewidth=linewidth)
plt.plot(epochsr2[0:20], loss2[0:20], 'go-', markersize=markersize, label='train loss with Aug.', linewidth=linewidth)
plt.plot(epochsr2[0:20], val_loss2[0:20], 'gs-', markersize=markersize, label='val loss with Aug.' ,linewidth=linewidth)
#plt.plot(epochsr3, loss3, 'yo-', markersize=markersize, label='Train(M-3)', linewidth=linewidth)
#plt.plot(epochsr3, val_loss3, 'ys-', markersize=markersize, label='Valid(M-3)', linewidth=linewidth)

plt.title('Training and Validation Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(prop={'size': 10}, loc=1)
plt.savefig('logs/loss_curve.svg')
plt.show()


sns.set(rc={'figure.figsize':(8,5)})
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)


plt.figure()
plt.plot(epochsr1[0:20], acc1[0:20], 'bo-', markersize=markersize, label='train acc.', linewidth=linewidth)
plt.plot(epochsr1[0:20], val_acc1[0:20], 'bs-', markersize=markersize, label='val acc', linewidth=linewidth)
plt.plot(epochsr2[0:20], acc2[0:20], 'go-', markersize=markersize, label='train acc. with Aug.', linewidth=linewidth)
plt.plot(epochsr2[0:20], val_acc2[0:20], 'gs-', markersize=markersize, label='val acc. with Aug.' ,linewidth=linewidth)
#plt.plot(epochsr3, acc3, 'yo-', markersize=markersize, label='Train(M-3)', linewidth=linewidth)
#plt.plot(epochsr3, val_acc3, 'ys-', markersize=markersize, label='Valid(M-3)', linewidth=linewidth)

plt.title('Training and Validation Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(prop={'size': 10}, loc=0)
plt.savefig('logs/acc_curve.svg')
plt.show()