import numpy as np

def myGaussianSmoothing(img,ker_size,std):
    kernel = gaussian_kernal(img,ker_size,std)
    return kernel    
def gaussian_kernal(img,ker_size,std):
    ker_1d = np.linspace(-(ker_size // 2), ker_size // 2, ker_size)
    for i in range(ker_size):
        ker_1d[i] = 1 / (np.sqrt(2 * np.pi) * std) * np.e ** (-np.power((ker_1d[i] - 0) / std, 2) / 2)
    trans = ker_1d.T
    ker_2d = np.outer(trans, trans)
    ker_2d *= 1.0 / ker_2d.max()
    
    conv = convolution(img,ker_2d)
    
    return conv

def convolution(img,ker_2d):
    res = np.zeros(img.shape)
    pad_height = int((ker_2d.shape[0] - 1) / 2)
    pad_width = int((ker_2d.shape[1] - 1) / 2)
    padded_image = np.zeros((img.shape[0] + (2 * pad_height), img.shape[1] + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = img
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            res[row, col] = np.sum(ker_2d * padded_image[row:row + ker_2d.shape[0], col:col + ker_2d.shape[1]])*2
            if True:
                res[row, col] /= ker_2d.shape[0] * ker_2d.shape[1]
    
    return res   
