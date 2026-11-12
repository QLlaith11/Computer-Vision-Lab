import numpy as np
import cv2
import matplotlib.pyplot as plt


def generate_gaussian_kernel(seg,siz):
    x=np.arange(-siz,siz)
    y=np.arange(-siz,siz)
    array=np.zeros((siz,siz))
    for i in x:
        for j in y:
            first=1/(2*np.pi*(seg**2))
            array[i][j]=first*(np.exp(-1*((i**2)+(j**2))/(2*(seg**2))))
    
    return array



# def convolution(img,mask,size):
#     r1,c1=img.shape[:2]
#     result= np.zeros((r1,c1))
#     for i in range(0,r1):
#         for j in range(0,c1):
#             if(i>=(r1-size)):
#                 break
#             elif (j>=(c1-size)):
#                 break
#             new=img[i:i+size,j:j+size]
#             value=np.multiply(mask,new)
#             new_value=np.sum(value)
#             result[i+(size//2)-1][j+(size//2)-1]=new_value
            
#     return result

# def convolution_value(img, mask, size):
#     r1, c1 = img.shape[:2]
#     total_value = 0 

#     for i in range(0, r1):
#         for j in range(0, c1):
#             if i >= (r1 - size) or j >= (c1 - size):
#                 continue 
            
#             new = img[i:i + size, j:j + size] 
#             value = mask * new
#             new_value = np.sum(value)
#             total_value += new_value

#     return total_value  


def convolution(img, mask, size):
    r, c = img.shape[:2]
    pad = size // 2  # Amount of padding needed to center kernel
    
    # Initialize result with zeros and padding to handle edges
    result = np.zeros((r, c))
    
    # Loop through each pixel
    for i in range(pad, r - pad):
        for j in range(pad, c - pad):
            # Extract the region of interest around the pixel
            part = img[i - pad:i + pad + 1, j - pad:j + pad + 1]
            
            # Apply element-wise multiplication and sum the result
            mul = np.multiply(mask, part)
            res = np.sum(mul)
            
            # Place the result at the correct position
            result[i, j] = res
    
    return result


# Step 1:
image = cv2.imread('C:\\Users\\IT2.student12\\Desktop\\Computer Vision\\lab6\\P0.jfif', cv2.IMREAD_GRAYSCALE)

# Step 2:
image_normalized = image / 255.0

# Step 3: Filter the image using Gaussian filter

gaussian_mask = generate_gaussian_kernel(1,3)
filtered_image=convolution(image_normalized,gaussian_mask,3)

# Step 4:
m1 = (1/175) * np.array([ [-13, 2, 7, 2, -13], [2, 17, 22, 17, 2], [7, 22, 27, 22, 7], [2, 17, 22, 17, 2], [-13, 2, 7, 2, -13] ])
m2 = (1/420) * np.array([ [31, -44, 0, 44, -31], [-5, -62, 0, 62, 5], [-17, -68, 0, 68, 17], [-5, -62, 0, 62, 5], [31, -44, 0, 44, -31] ])
m3 = (1/420) * np.array([ [-31, 5, 17, 5, -31], [44, 62, 68, 62, 44], [0, 0, 0, 0, 0], [-44, -62, -68, -62, -44], [31, -5, -17, -5, 31] ])
m4= (1/70) * np.array([ [2, -1, -2, -1, 2], [2, -1, -2, -1, 2], [2, -1, -2, -1, 2], [2, -1, -2, -1, 2], [2, -1, -2, -1, 2] ])
m5 = (1/100) * np.array([ [-4, -2, 0, 2, 4], [-2, -1, 0, 1, 2], [0, 0, 0, 0, 0], [2, 1, 0, -1, -2], [4, 2, 0, -2, -4] ])
m6 = (1/70) * np.array([ [2, 2, 2, 2, 2], [-1, -1, -1, -1, -1], [-2, -2, -2, -2, -2], [-1, -1, -1, -1, -1], [2, 2, 2, 2, 2] ])
m7 = (1/60) * np.array([ [-1, 2, 0, -2, 1], [-1, 2, 0, -2, 1], [-1, 2, 0, -2, 1], [-1, 2, 0, -2, 1], [-1, 2, 0, -2, 1] ])
m8 = (1/140) * np.array([ [4, -2, -4, -2, 4], [2, -1, -2, -1, 2], [0, 0, 0, 0, 0], [-2, 1, 2, 1, -2], [-4, 2, 4, 2, -4] ])
m9 = (1/140) * np.array([ [-4, -2, 0, 2, 4], [2, 1, 0, -1, -2], [4, 2, 0, -2, -4], [2, 1, 0, -1, -2], [-4, -2, 0, 2, 4] ])
m10 = (1/60) * np.array([ [1, 1, 1, 1, 1], [-2, -2, -2, -2, -2], [0, 0, 0, 0, 0], [2, 2, 2, 2, 2], [-1, -1, -1, -1, -1] ])

masks=[m1,m2,m3,m4,m5,m6,m7,m8,m9,m10]
# Step 5:
rho0 = 0.005

# k_values = np.zeros(10)
result_image = np.zeros_like(image_normalized)
# k_values=[]
for i in range(2, filtered_image.shape[0] - 3):
    for j in range(2, filtered_image.shape[1] - 3):
    
        local_window = filtered_image[i-2:i+3, j-2:j+3]
        

        k1=convolution(local_window, m1, 5)
        k2=convolution(local_window, m2, 5)
        k3=convolution(local_window, m3, 5)
        k4=convolution(local_window, m4, 5)
        k5=convolution(local_window, m5, 5)
        k6=convolution(local_window, m6, 5)
        k7=convolution(local_window, m7, 5)
        k8=convolution(local_window, m8, 5)
        k9=convolution(local_window, m9, 5)
        k10=convolution(local_window, m10, 5)
       

        sin_th=k3/np.sqrt((k2**2)+(k3**2))
        cos_th=k2/np.sqrt((k2**2)+(k3**2))

        if (sin_th is not np.NaN) and (cos_th is not np.NaN):
            c2=k4*(cos_th**2)+k5*sin_th*cos_th+ k6*(sin_th**2)
            c3=k7*(cos_th**3)+k8*sin_th*(cos_th**2)+k9*(sin_th**2)*cos_th+ k10*(sin_th**3)
            if c3 < 0 and abs(c2 / (3 * c3)) > rho0:
                result_image[i, j] = 255


    

plt.subplot(1,2,1)
plt.imshow(filtered_image,cmap='gray')
plt.title('filtered_image')
plt.subplot(1,2,2)
plt.imshow(result_image,cmap='gray')
plt.title('result_image')
plt.show()