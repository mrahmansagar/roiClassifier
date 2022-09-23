# -*- coding: utf-8 -*-
"""
Created on Wed Sept. 10 16:47:49 2022

@author: Mmr Sagar
PhD Student | AG Alves 
MPI for Multidisciplinary Sciences 
=====================================

Utility functions for preprocessing the data and training the network 
"""

# Required libraries 
import numpy as np
import ctypes
import matplotlib.pyplot as plt


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



def norm8bit(v, minVal=None, maxVal=None):
    """
    NORM8BIT function takes an array and normalized it before converting it into 
    a 8 bit unsigned integer and returns it.

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
    numpy.ndarray (uint8)
        Numpy Array of same dimension as input with data type as unsigned integer 8 bit

    """
    if minVal == None:
        minVal = v.min()
    
    if maxVal == None:
        maxVal = v.max()
      
    maxVal -= minVal
      
    v = ((v - minVal)/maxVal) * 255
    
    return v.astype(np.uint8)


