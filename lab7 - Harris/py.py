import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def gaussian_kernel(size, sig):
	kernel = np.zeros((size,size))
	center = size // 2
	
	for i in range(size):
		for j in range(size):
			x = i - center
			y = j - center
			kernel[i,j] = 1/(2* np.pi * sig**2)*np.exp(-(x**2 + y**2)/(2 * sig**2))
	kernel /= np.sum(kernel)
	return kernel
def conv(img, mask):
	h,w = img.shape
	mh, mw = mask.shape
	pad_h = mh//2
	pad_w = mw//2
	im_h = h + pad_h
	im_w = w + pad_w
	padded = np.zeros((im_h, im_w))
	result = np.zeros_like(img)
	for i in range(h):
		for j in range(w):
			padded[i+pad_h,j+pad_w] = img[i,j]
	
	for i in range(h):
		for j in range(w):
			region = []
			for x in range(mh):
				row = []
				for y in range(mw):
					row.append(padded[i+x,i+y])
				region.append(row)
			value = 0
			for x in range(mh):
				for y in range(mw):
					value += region[x,y] * mask[x,y]
			result[i,j] = value
	return result
					
					
					
					
					
					
					
					
					
					
