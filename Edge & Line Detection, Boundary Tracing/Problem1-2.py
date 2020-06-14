# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 03:59:20 2019

@author: yashd
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import cv2 


def appendList(i, j, x, y):
    
    for m in range(len(i)):
        Boundary_list.append([x+i[m], y+j[m]])


if __name__ == "__main__":

    Boundary_list = []
    img = cv2.imread('1.png')
    img2 = cv2.imread('1.png',0)
    img2_copy = img2.copy()
   
    height, width = img2.shape
    smoothImage = ndi.median_filter(img2, size=9)
    sobelx = cv2.Sobel(smoothImage,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(smoothImage,cv2.CV_64F,0,1,ksize=5)
    magnitude =  cv2.Canny(smoothImage,100,200)
    
    kernel = 11
   
    thresh = np.mean(magnitude)
      
    for x in range(0 ,height-kernel, kernel):
        for y in range(0 ,width-kernel, kernel):
            window = magnitude[x:x+kernel, y:y+kernel]
            max_val=np.max(window)
            
           
            i, j = np.where(window == max_val)
            
            window[:] = 0
            
            if max_val >= thresh:
                window[i, j] = 1   
                appendList(i, j, x, y)   

            magnitude[x:x+kernel, y:y+kernel] = window

    Boundary_list = np.array(Boundary_list)
    
    fig, axs = plt.subplots()
    axs.set_title('Boundary Tracing')
    axs.imshow(img2, cmap='gray')
    plt.scatter(B_list[:, 1], B_list[:, 0], c='b', s=0.4)
    plt.show()
