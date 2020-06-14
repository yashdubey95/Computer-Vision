# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 20:50:45 2019

@author: yashd
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


def display_img(image):
    plt.imshow(image, cmap='gray')
    plt.show()

def myHoughLine(img_bin, n):
    row = img_bin.shape[0]
    col = img_bin.shape[1]
    theta_range = np.linspace(-90.0, 0.0, np.ceil(90.0) + 1.0)
    theta = np.concatenate((theta_range, -theta_range[len(theta_range)-2::-1]))

    dist = np.ceil((np.sqrt((row - 1)**2 + (col - 1)**2)))
    rho = np.linspace(-dist, dist, 2*dist + 1)
    H = np.zeros((len(rho), len(theta)))
    for i in range(row):
        for j in range(col):
            if img_bin[i, j]:
                for thIdx in range(len(theta)):
                    rhoVal = j*np.cos(theta[thIdx]*np.pi/180.0) + i*np.sin(theta[thIdx]*np.pi/180)
                    rhoIdx = np.nonzero(np.abs(rho-rhoVal) == np.min(np.abs(rho-rhoVal)))[0]
                    H[rhoIdx[0], thIdx] += 1
    
    lines = topNLines(H,n,rho,theta)
    
    return lines

def topNLines(ht_acc_matrix, n, rhos, thetas):

  flat = sorted(list(set(np.hstack(ht_acc_matrix))), key = lambda n: -n)
  coords_sorted = [(np.argwhere(ht_acc_matrix == acc_value)) for acc_value in flat[0:n]]
  rho_theta = []
  for coords_for_val_idx in range(0, len(coords_sorted), 1):
    coords_for_val = coords_sorted[coords_for_val_idx]
    for i in range(0, len(coords_for_val), 1):
      n = coords_for_val[i][0]
      m = coords_for_val[i][1]
      rho_theta.append([rhos[n], thetas[m]])
  
  return rho_theta[0:n]

def plottable(xmax, ymax, pt):

  x, y = pt
  flag = 1 if x <= xmax and x >= 0 and y <= ymax and y >= 0 else 0
  if flag == 1:
      return True
  else:
      return False

def draw_lines(img, pairs):

  target_im = img
  im_y_max, im_x_max, channels = np.shape(target_im)
  for i in range(0, len(pairs), 1):
    rho, theta = pairs[i][0], pairs[i][1] * np.pi / 180
    m = -np.cos(theta) / np.sin(theta)
    b = rho / np.sin(theta)
    left, right, top, bottom = (0, b), (im_x_max, im_x_max * m + b), (-b / m, 0), ((im_y_max - b) / m, im_y_max)
    pts = [pt for pt in [left, right, top, bottom] if plottable( im_x_max, im_y_max, pt)]
    if len(pts) == 2:
        x1 = round(pts[0][0])
        y1 = round(pts[0][1])
        x2 = round(pts[1][0])
        y2 = round(pts[1][1])            
        cv2.line(target_im, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 1)
      
  return target_im

if __name__ == '__main__':
    
    img = cv2.imread('3.png')
    img1 = cv2.imread('3.png',0)
    imBW = cv2.Canny(img1, threshold1 = 0, threshold2 = 50, apertureSize = 3)
    lines = myHoughLine(imBW, 5)
    final_image = draw_lines(img, lines)
    display_img(final_image)
