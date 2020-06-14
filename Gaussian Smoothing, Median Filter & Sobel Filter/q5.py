# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 20:51:11 2019

@author: yashd
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import Gaussian_smoothing
from PIL import Image

def sobel_edge_detection(image, sb):
    new_image_x = Gaussian_smoothing.convolution(image, sb)
    new_image_y = Gaussian_smoothing.convolution(image, np.flip(sb.T, axis=0))
    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    gradient_magnitude *= 1.0 / gradient_magnitude.max()
    ori = np.arctan2(new_image_x,new_image_y)*180/np.pi
    return gradient_magnitude, ori

def display_edge(image):
    plt.imshow(image, cmap='gray')
    plt.show()
        
if __name__ == '__main__':
    sobelFilter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    img = cv2.imread('Lena.png',0) 
    img2 = img / 255.0
    plt.imshow(img2,cmap='gray')
    image = Gaussian_smoothing.myGaussianSmoothing(img2,11,1)
    mag,ori = sobel_edge_detection(image, sobelFilter)
    display_edge(mag)
    display_edge(ori)
    H = ori
    S = mag
    V = mag
    HSVArray = np.zeros((512,512,3), 'uint8')
    HSVArray[..., 0] = H
    HSVArray[..., 1] = S
    HSVArray[..., 2] = V
    HSV = Image.fromarray(HSVArray)
    plt.imshow(HSV)
#    final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#    display_edge(final)
    