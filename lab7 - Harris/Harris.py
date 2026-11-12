
import numpy as np
import cv2
import matplotlib.pyplot as plt


def generate_gaussian_kernel(sig, siz):
    
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
 
        
def calculate_gradients(image):
    Hx = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    Hy = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    Ix = convolution(image, Hx)
    Iy = convolution(image, Hy)
    
    return Ix, Iy
        
        
        
def harris_corner_detection(gray_image, sigma1, size1,sigma2,size2, threethold):
    gaussian_filter=generate_gaussian_kernel(sigma1,size1)
    print(gaussian_filter.round(4))
    smoothed_image=convolution(gray_image,gaussian_filter)
    
    #question2 
    Ix,Iy=calculate_gradients(smoothed_image)
    
        
    #question3
    Ixx=Ix*Ix
    Iyy=Iy*Iy
    Ixy=Ix*Iy

    #question4
    gaussian_filter_large = generate_gaussian_kernel(sigma2, size2).round(4)
    # print(gaussian_filter_large)

    Mxx = convolution(Ixx, gaussian_filter_large)
    Myy = convolution(Iyy, gaussian_filter_large)
    Mxy = convolution(Ixy, gaussian_filter_large)
    
    

    # Question 5: Compute Harris response

    if Mxx is None:
        Mxx=0
    if Myy is None:
        Myy=0
    if Mxy is None:
        Mxy=0
        
    detM = (Mxx * Myy) - (Mxy ** 2)
    traceM = Mxx + Myy
    R = detM /traceM
    
    
    #question 6

    nms = np.zeros_like(R)
    window_size = 10
    offset = window_size // 2

    for i in range(offset, R.shape[0] - offset):
        for j in range(offset, R.shape[1] - offset):
            window = R[i-offset:i+offset+1, j-offset:j+offset+1]
            if R[i, j] >= np.max(window):
                nms[i, j] = R[i, j]
    nms = nms / nms.max()
    corners1 = (nms > threethold)
    # corners2 = (nms> threethild2)
    
    return corners1
        
        

#main
image_path="photo_2025-01-09_13-25-38.jpg"
gray_image=cv2.imread(image_path,0)
image=cv2.imread(image_path,1)
if gray_image is None:
    print("Error: the image could not be loaded.")
    exit()

corners = harris_corner_detection(gray_image, 1, 5, 2, 13,0.35)


plt.imshow(gray_image, cmap='gray')
plt.scatter(np.nonzero(corners)[1], np.nonzero(corners)[0], c='red', s=5)
plt.title("Harris Corners")
plt.waitforbuttonpress(timeout=-1)

# plt.figure()
# plt.imshow(gray_image, cmap='gray')
# plt.scatter(np.nonzero(corners2)[1], np.nonzero(corners2)[0], c='red', s=5)
# plt.title("Harris Corners2")
# plt.show()













    

