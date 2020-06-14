# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:26:48 2019

@author: yashd
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import Gaussian_smoothing

def downsample(image,img):
    sample_down = image
    down_1 = sample_down[0::2,0::2]
    down_2 = down_1[0::2,0::2]
    down = cv2.resize(down_2, img.shape, interpolation = cv2.INTER_AREA)
    return down,down_2

def upsample(image,img):
    out1 = np.zeros((image.shape[0]*2,image.shape[1]*2),dtype=image.dtype)
    out1[::2,::2] = image
    out1_g = out1
    out1_m = out1
    up1_smooth = Gaussian_smoothing.myGaussianSmoothing(out1_g,11,1)
    up1_median = myMedianFilter(out1_m,11)
    out2 = np.zeros((up1_smooth.shape[0]*2,up1_smooth.shape[1]*2),dtype=image.dtype)
    out3 = np.zeros((up1_median.shape[0]*2,up1_median.shape[1]*2),dtype=image.dtype)
    out2[::2,::2] = up1_smooth
    out3[::2,::2] = up1_median
    up2_smooth = Gaussian_smoothing.myGaussianSmoothing(out2,11,1)
    up2_median = myMedianFilter(out3,11)
    return up2_smooth, up2_median
    

def display_img(image):
    plt.imshow(image, cmap='gray')
    plt.show()

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
    
if __name__ == '__main__':

    img = cv2.imread('Lena.png',0) 
    display_img(img)
    img2 = img / 255.0
    display_img(img2)
    down,down2 = downsample(img2,img)
    display_img(down)
    up_smooth, up_median = upsample(down2,img2)

    display_img(up_smooth)
    display_img(up_median)