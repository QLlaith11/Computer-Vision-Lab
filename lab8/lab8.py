import numpy as np
import cv2
import matplotlib.pyplot as plt



def generate_gaussian_kernel(sig, siz):

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
 

def find_local_extrema(octave):
    local_maxima = []
    local_minima = []
    
    for i in range(1, len(octave) - 1):
        mid_level = octave[i]
        upper_level = octave[i - 1]
        lower_level = octave[i + 1]
        
        for x in range(1, mid_level.shape[0] - 1):
            for y in range(1, mid_level.shape[1] - 1):
                current_pixel = mid_level[x, y]


                window_upper = upper_level[x-1:x+2, y-1:y+2]
                window_mid = mid_level[x-1:x+2, y-1:y+2]
                window_lower = lower_level[x-1:x+2, y-1:y+2]

                #current pixel is a local maximum or minimum
                if (current_pixel > np.max(window_upper) and 
                    current_pixel > np.max(window_mid) and 
                    current_pixel > np.max(window_lower)):
                    local_maxima.append((x, y, i))  # Local maximum

                elif (current_pixel < np.min(window_upper) and 
                      current_pixel < np.min(window_mid) and 
                      current_pixel < np.min(window_lower)):
                    local_minima.append((x, y, i))  # Local minimum

    return local_maxima, local_minima

def multiply_extrema_with_dog(octave, local_maxima, local_minima):
 
    modified_maxima_image = octave[1].copy()  
    modified_minima_image = octave[1].copy()

  
    for (x, y, level) in local_maxima:
        modified_maxima_image[x, y] *= octave[level][x, y]  

    for (x, y, level) in local_minima:
        modified_minima_image[x, y] *= octave[level][x, y]  

    return modified_maxima_image, modified_minima_image




#Q1
image_path="I0.jpg"
image=cv2.imread(image_path,0)
if image is None:
    print("Error: the image could not be loaded.")
    exit()

plt.imshow(image)
plt.show()

#Q2_3
img_resized_400=cv2.resize(image,(400,400))
img_resized_200=cv2.resize(image,(200,200))
img_resized_100=cv2.resize(image,(100,100))
img_resized_50=cv2.resize(image,(50,50))

#Q4
size1 = 400
size2= 200
size3 =100
size4=50

sigma1= 2
sigma2= np.sqrt(2)*sigma1
sigma3= np.sqrt(2)*sigma2
sigma4= np.sqrt(2)*sigma3

# print(6*sigma1)

G1= generate_gaussian_kernel(sigma1, 11)
G2= generate_gaussian_kernel(sigma2, 17)
G3= generate_gaussian_kernel(sigma3, 23)
G4= generate_gaussian_kernel(sigma4, 33)

IG_array=[img_resized_400, img_resized_200, img_resized_100, img_resized_50]
G_array=[G1,G2,G3,G4]

octaves = []

for i in range(len(IG_array)):
    octave = []  
    for j in range(len(G_array) - 1):  
        convolved1 = convolution(IG_array[i], G_array[j])
        convolved2 = convolution(IG_array[i], G_array[j + 1])
        dog = convolved1 - convolved2
        octave.append(dog)
    octaves.append(octave)

# plt.imshow(octaves[0][0], cmap='gray')
# plt.title("First DoG in the first octave")
# plt.show()


all_local_maxima = []
all_local_minima = []

for octave in octaves:
    local_maxima, local_minima = find_local_extrema(octave)
    all_local_maxima.append(local_maxima)
    all_local_minima.append(local_minima)

local_maxima_first_octave = all_local_maxima[0]
local_minima_first_octave = all_local_minima[0]

print("Local Maxima (first octave):", local_maxima_first_octave)
print("Local Minima (first octave):", local_minima_first_octave)

plt.imshow(octaves[0][1], cmap='gray')
for (x, y, level) in local_maxima_first_octave:
    plt.scatter(y, x, color='red', marker='x', label='Maxima' if level == 0 else "")
for (x, y, level) in local_minima_first_octave:
    plt.scatter(y, x, color='blue', marker='o', label='Minima' if level == 0 else "")
plt.title("Local Extrema on the First DoG in First Octave")
plt.show()




modified_maxima_image, modified_minima_image = multiply_extrema_with_dog(octaves[0], local_maxima_first_octave, local_minima_first_octave)


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(modified_maxima_image, cmap='gray')
plt.title("Modified Image with Local Maxima Multiplied by DOG")

plt.subplot(1, 2, 2)
plt.imshow(modified_minima_image, cmap='gray')
plt.title("Modified Image with Local Minima Multiplied by DOG")

plt.show()
