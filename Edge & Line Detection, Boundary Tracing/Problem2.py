# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 03:59:20 2019

@author: yashd
"""
import cv2
import numpy as np
import Gaussian_smoothing


def findCorners(img, window_size, k, thresh):
    sobelFilter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    I_x = Gaussian_smoothing.convolution(img, sobelFilter)
    I_y = Gaussian_smoothing.convolution(img, np.flip(sobelFilter.T, axis=0))
    I_xx, I_xy, I_yy = I_x**2, I_x*I_y, I_y**2
    height = img.shape[0]
    width = img.shape[1]

    newImg = img.copy()
    color_img = cv2.cvtColor(newImg, cv2.COLOR_GRAY2RGB)
    offset = window_size//2

    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            windowIxx = I_xx[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIxy = I_xy[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIyy = I_yy[y-offset:y+offset+1, x-offset:x+offset+1]
            Sxx = np.sum(windowIxx)
            Sxy = np.sum(windowIxy)
            Syy = np.sum(windowIyy)
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            r = det - k*(trace**2)
            if r > thresh:
                color_img.itemset((y, x, 0), 0)
                color_img.itemset((y, x, 1), 0)
                color_img.itemset((y, x, 2), 255)
    return color_img

if __name__ == "__main__":

    window_size = 5
    k = 0.04
    thresh = 1000000000
    img = cv2.imread('2-1.jpg')
    img1 = cv2.imread('2-1.jpg',0)
    
    imgg = cv2.imread('2-2.jpg')
    imgg1 = cv2.imread('2-2.jpg',0)    
    
    
    finalImg1 = findCorners(img1, int(window_size), float(k), int(thresh))
    if finalImg1 is not None:
        cv2.imwrite("Pic1.png", finalImg1)

    finalImg2 = findCorners(imgg1, int(window_size), float(k), int(thresh))
    if finalImg2 is not None:
        cv2.imwrite("Pic2.png", finalImg2)
