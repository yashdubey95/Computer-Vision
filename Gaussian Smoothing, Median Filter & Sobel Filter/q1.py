import cv2
import numpy as np
import matplotlib.pyplot as plt

def downsample(image,img):
    sample_down = image
    down_1 = sample_down[0::2,0::2]
    down_2 = down_1[0::2,0::2]
    down = cv2.resize(down_2, img.shape, interpolation = cv2.INTER_AREA)
    return down,down_2

def upsample(image,img):
    out1 = np.zeros((image.shape[0]*2,image.shape[1]*2),dtype=image.dtype)
    out1[::2,::2] = image 
    out2 = np.zeros((out1.shape[0]*2,out1.shape[1]*2),dtype=image.dtype)
    out2[::2,::2] = out1
    return out2    

def display_img(image):
    plt.imshow(image, cmap='gray')
    plt.show()
   
if __name__ == '__main__':

    img = cv2.imread('Lena.png',0) 
    display_img(img)
    img2 = img / 255.0
    display_img(img2)
    down,down2 = downsample(img2,img)
    display_img(down)
    up = upsample(down2,img)
    display_img(up)
