# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:04:54 2019

@author: yashd
"""
import numpy as np

def myMedianFilter(mf_img,k):
   f_window = filterWindow(mf_img,k)
   return f_window

def filterWindow(img,k):
   mf_out = np.zeros((img.shape[0],img.shape[1]))
   window = np.zeros(k*k)
   edgex = (k // 2)
   edgey = (k // 2)
   for x in range(edgex, img.shape[0] - edgex):
       for y in range(edgey, img.shape[1] - edgey):
           i = 0
           for fx in range(0, k):
               for fy in range(0, k):
                   window[i] = img[x + fx - edgex][y + fy - edgey]
                   i = i + 1
           window.sort()
           mf_out[x][y] = np.median(window)
   
   return mf_out