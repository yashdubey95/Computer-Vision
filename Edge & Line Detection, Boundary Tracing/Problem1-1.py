# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 17:22:35 2019

@author: yashd
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import Gaussian_smoothing
import sobel


def display_img(image):
    plt.imshow(image, cmap='gray')
    plt.show()

def display_quiver(image):
    plt.quiver(image)
    plt.show()
    
if __name__ == '__main__':
    img1 = cv2.imread('1.png')
    img = cv2.imread('1.png',0)
    img2 = img / 255.0
    up1_smooth = Gaussian_smoothing.myGaussianSmoothing(img2,11,1)
    sobelFilter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    mag,Ix, Iy = sobel.sobel_edge_detection(up1_smooth, sobelFilter)
    display_img(mag)
    fig,axs = plt.subplots()
    plt.imshow(mag,cmap='gray')
    axs.quiver(Ix,Iy, scale = 1.5, color = 'r')
    plt.show()
