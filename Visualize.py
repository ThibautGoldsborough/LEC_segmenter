#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:17:13 2019

@author: thibautgold
"""

import os, sys
import numpy as np
import cv2 as cv
from skimage.morphology import skeletonize
from random import randrange
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
import pickle as pickle
import easygui as gui
from xlwt import Workbook 


with open('SAVED_WORK', 'rb') as infile:
    outcome = pickle.load(infile) 

def display(outcome):
    cv.namedWindow('Press E to Exit')  
    frame_num=0  
    while(1):
        k = cv.waitKey(1) & 0xFF
        display_right=outcome[frame_num][0]
        display_left=outcome[frame_num][3]
        display= np.hstack((display_left,display_right))
        cv.imshow('Press E to Exit',display)
        cv.moveWindow('Press E to Exit',150,10)
        if k ==ord('p'):
            if frame_num<(len(outcome)-1):
                frame_num+=1
        if k==ord('o'):
            if frame_num>0:
                frame_num-=1              
        if k==ord('e'):
            break              
        
        
display(outcome)