import numpy as np
import matplotlib.pyplot as plt
import cv2


# question 1
def generate_gaussian_kernel(sig, siz):
    # Ensure the kernel size is odd
    assert siz % 2 == 1, "Kernel size must be odd."
    
    gaussian_kernel = np.zeros((siz, siz))
    center = siz // 2
 
    for i in range(siz):
        for j in range(siz):
            x = i - center
            y = j - center
            gaussian_kernel[i, j] = (1 / (2 * np.pi * sig**2)) * np.exp(-(x**2 + y**2) / (2 * sig**2))
    # # Normalize the kernel to ensure the sum is 1
    # gaussian_kernel /= np.sum(gaussian_kernel)

    return gaussian_kernel
            
# queston 2

                 
def convolution(img, mask):
 
    img_h,img_w = img.shape
    mask_h, mask_w = mask.shape

    assert mask_h % 2 == 1 and mask_w % 2 == 1, "Kernel dimensions must be odd."

    pad_h = mask_h // 2
    pad_w = mask_w // 2

    padded_img_h = img_h + 2 * pad_h
    padded_img_w = img_w + 2 * pad_w
    padded_img = np.zeros((padded_img_h, padded_img_w))

    for i in range(img_h):
        for j in range(img_w):
            padded_img[i + pad_h][j + pad_w] = img[i][j]

    result = np.zeros((img_h, img_w))

    for i in range(img_h):
        for j in range(img_w):
            region = []
            for x in range(mask_h):
                row = []
                for y in range(mask_w):
                    row.append(padded_img[i + x][j + y])
                region.append(row)

            value = 0
            for x in range(mask_h):
                for y in range(mask_w):
                    value += region[x][y] * mask[x][y]

            result[i, j] = value

    return result


def sort2d(square_array):
    values =list()
    for i in range(len(square_array)):
        for j in range(len(square_array)):
            values.append(square_array[i,j])
        values=np.sort()


# question 3
image_path="F:\\University\\Fourth year\\Computer Vision\\lab3\\noisy.png"
pic=cv2.imread (image_path, cv2.IMREAD_GRAYSCALE)
print("gray image is: ", pic)

#question 4
k1=(1/9)*np.ones((3,3))
k2=generate_gaussian_kernel(3,7)


a=convolution(pic,k1)  


a2=convolution(pic,k2)

plt.subplot(1,3,1)
plt.imshow(pic,cmap='gray')
plt.subplot(1,3,2)
plt.imshow(a,cmap='gray')
plt.subplot(1,3,3)
plt.imshow(a2,cmap='gray')
plt.show()




