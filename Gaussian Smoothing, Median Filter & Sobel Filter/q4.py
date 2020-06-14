# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:56:55 2019

@author: yashd
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import Gaussian_smoothing
import Median_Filter

def display_img(image):
    plt.imshow(image, cmap='gray')
    plt.show()


if __name__ == '__main__':
    
    img = cv2.imread('Lena.png',0) 
    img2 = img / 255.0
    display_img(img2)
    noisy_img = img2 + np.random.normal(0, 0.1, img.shape)
    diff = noisy_img - img2
    display_img(noisy_img)
    noisy_img_g = noisy_img
    noisy_img_m = noisy_img
    noisy_smooth = Gaussian_smoothing.myGaussianSmoothing(noisy_img_g,11,1)   
    display_img(noisy_smooth)
    noisy_median = Median_Filter.myMedianFilter(noisy_img_m,7)
    display_img(noisy_median)
    noisy_img_changed = img2 + np.where(np.random.normal(0,0.1,img.shape)>0.2,1,0)
    display_img(noisy_img_changed)
    noisy_img_changed_g = noisy_img_changed
    noisy_img_changed_m = noisy_img_changed
    noisy_smooth_c = Gaussian_smoothing.myGaussianSmoothing(noisy_img_changed_g,11,1)
    display_img(noisy_smooth_c)
    noisy_median_c = Median_Filter.myMedianFilter(noisy_img_changed_m,7)
    display_img(noisy_median_c)
    
    