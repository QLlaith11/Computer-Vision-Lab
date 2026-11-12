import cv2
import numpy as np
import matplotlib.pyplot as plt



def nonLinearFilter(img, size):
    r, c = img.shape[:2]
    pad = size // 2 
    
  
    Min = np.zeros((r, c))
    Max = np.zeros((r, c))
    Med = np.zeros((r, c))
    
    for i in range(pad, r - pad):
        for j in range(pad, c - pad):
         
            part = img[i - pad:i + pad + 1, j - pad:j + pad + 1]

            [max_val, min_val, median] = sort2d (part)
            
            Min[i, j] = max_val
            Max[i, j] = min_val
            Med[i, j] = median

    
    return Min,Max,Med


def sort2d(array):
    vector = [item for row in array for item in row]
    
    # bubble sort
    n = len(vector)
    for i in range(n):
        for j in range(0, n - i - 1):
            if vector[j] > vector[j + 1]:
                vector[j], vector[j + 1] = vector[j + 1], vector[j]
    
   
    min_val = vector[0]
    max_val = vector[-1]

    mid = n // 2
    if n % 2 == 0:
        median = (vector[mid - 1] + vector[mid]) / 2
    else:
        median = vector[mid]
    
    return max_val, min_val, median



size = 3

image_path="F:\\University\\Fourth year\\Computer Vision\\lab4\\P.png"
img = cv2.imread (image_path,0)

Max,Min,Med = nonLinearFilter(img, size)

plt.subplot(1,4,1)
plt.title ("The Original Image")
plt.imshow(img, cmap = 'gray')


plt.subplot(1,4,2)
plt.title ("The Filtered Image (Minimum)")
plt.imshow(Min, cmap = 'gray')

plt.subplot(1,4,3)
plt.title ("The Filtered Image (Maximum)")
plt.imshow(Max, cmap = 'gray')

plt.subplot(1,4,4)
plt.title ("The Filtered Image (Median)")
plt.imshow(Med, cmap = 'gray')

plt.show()
