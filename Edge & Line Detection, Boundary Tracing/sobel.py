# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 20:01:09 2019

@author: yashd
"""
import Gaussian_smoothing
import numpy as np

def sobel_edge_detection(image, sb):
    new_image_x = Gaussian_smoothing.convolution(image, sb)
    new_image_y = Gaussian_smoothing.convolution(image, np.flip(sb.T, axis=0))
    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    gradient_magnitude *= 1.0 / gradient_magnitude.max()
    ori = np.arctan2(new_image_x,new_image_y)*180/np.pi
    return gradient_magnitude, new_image_x, new_image_y