import numpy as np
import matplotlib.pyplot as plt
from math import atan2, degrees, sqrt
import cv2


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
    gaussian_kernel /= np.sum(gaussian_kernel)
    return gaussian_kernel
               
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
                 

#quesion 1
path1="F:\\University\\Fourth year\\Computer Vision\\lab5\\city.jpg"
img=cv2.imread(path1,0)
# print('img= ',img.shape)
img_resized=cv2.resize(img,(200,200))
# print('img_resized= ',img_resized.shape)

#quesion 2

seg=1
siz=5
G=generate_gaussian_kernel(seg,siz)

img_remove_noise=convolution(img_resized,G)

# fig,ax
plt.figure()
plt.subplot(3,2,1)
plt.imshow(img_resized,cmap='gray')
plt.title('img_resized')
plt.subplot(3,2,2)
plt.imshow(img_remove_noise,cmap='gray')
plt.title('img_remove_noise')
# plt.show()

#question 3
def sobel_edges(image):
    Hx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Hy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Ix = convolution(image, Hx)
    Iy = convolution(image, Hy)

    magnitude = np.sqrt(Ix**2 + Iy**2)
    direction = np.zeros_like(magnitude)
    
    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            direction[i, j] = degrees(atan2(Iy[i, j], Ix[i, j])) % 360

    # Normalize magnitude to range [0, 1]
    magnitude = magnitude / np.max(magnitude)
    
    return Ix, Iy, magnitude, direction

Ix,Iy,magnitude,direction=sobel_edges(img_remove_noise)

# print("Ix= ",Ix)
# print("Iy= ",Iy)
# print("magnitude= ",magnitude)
# print("direction= ",direction)

plt.subplot(3,2,3)
plt.title('magnitiude')
plt.imshow(magnitude,cmap='gray')
# plt.show()

#question 4
def quantize_angles(direction):
    quantized = np.zeros_like(direction)
    for i in range(direction.shape[0]):
        for j in range(direction.shape[1]):
            angle = direction[i, j]
 
            if 0 <= angle < 22.5 or 337.5 <= angle <= 360:
                quantized[i, j] = 0
            elif 22.5 <= angle < 67.5:
                quantized[i, j] = 45
            elif 67.5 <= angle < 112.5:
                quantized[i, j] = 90
            elif 112.5 <= angle < 157.5:
                quantized[i, j] = 135
            elif 157.5 <= angle < 202.5:
                quantized[i, j] = 180
            elif 202.5 <= angle < 247.5:
                quantized[i, j] = 225
            elif 247.5 <= angle < 292.5:
                quantized[i, j] = 270
            elif 292.5 <= angle < 337.5:
                quantized[i, j] = 315
    
    return quantized

# quantized_angles=quantize_angles(direction)
# print("quantize_angles= ",quantized_angles)


# question 5
def non_maximum_suppression(magnitude, direction):
    suppressed = np.zeros_like(magnitude)
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            angle = direction[i, j]
            q, r = 0, 0
            if angle == 0 or angle == 180:
                q, r = magnitude[i, j + 1], magnitude[i, j - 1]
            elif angle == 45 or angle == 225:
                q, r = magnitude[i + 1, j - 1], magnitude[i - 1, j + 1]
            elif angle == 90 or angle == 270:
                q, r = magnitude[i + 1, j], magnitude[i - 1, j]
            elif angle == 135 or angle == 315:
                q, r = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]

            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                suppressed[i, j] = magnitude[i, j]
            else:
                suppressed[i, j] = 0  
    return suppressed

Eliminated=non_maximum_suppression(magnitude,direction)
# print("Eliminated= ",Eliminated)

plt.subplot(3,2,4)
plt.title('Eliminated')
plt.imshow(Eliminated,cmap='gray')


#question 6
def classify_edges(suppressed_org, low_threshold, high_threshold):

    suppressed = np.copy(suppressed_org)

    strong = suppressed >= high_threshold
    weak = suppressed < low_threshold

    candidate = (suppressed >= low_threshold) & ~strong & ~weak

    suppressed[weak] = 0

    for i in range(1, suppressed.shape[0] - 1):
        for j in range(1, suppressed.shape[1] - 1):
            if candidate[i, j]:
                # Manually check the 3x3 neighborhood for strong pixels
                has_strong_neighbor = False
                for x in range(-1, 2):  # -1, 0, 1
                    for y in range(-1, 2):  # -1, 0, 1
                        if strong[i + x, j + y]:
                            has_strong_neighbor = True
                            break
                    if has_strong_neighbor:
                        break
                    
                if not has_strong_neighbor:
                    suppressed[i, j] = 0

    return suppressed



classify_edges=classify_edges(Eliminated,0.03,0.3)
# plt.figure()
plt.subplot(3,2,5)
plt.imshow(classify_edges,cmap='gray')
plt.title('classify_edges')
plt.show()
























# # Main Task Execution
# def canny_from_scratch(image, low_threshold, high_threshold):
#     resized = resize_image(image)
#     blurred = gaussian_blur(resized)
#     Ix, Iy, magnitude, direction = sobel_edges(blurred)
#     quantized = quantize_angles(direction)
#     suppressed = non_maximum_suppression(magnitude, quantized)
#     edges = double_threshold(suppressed, low_threshold, high_threshold)
#     final_edges = hysteresis(edges)
#     return final_edges

# # Load your image
# path = "F:\\University\\Fourth year\\Computer Vision\\lab5\\P.png"
# image = plt.imread(path)
# if image.ndim == 3:  # Convert RGB to grayscale
#     image = grayscale(image)

# final_edges = canny_from_scratch(image, low_threshold=0.03, high_threshold=0.3)

# # Display Results
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(image, cmap='gray')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title('Canny Edges')
# plt.imshow(final_edges, cmap='gray')
# plt.axis('off')
# plt.show()
