import cv2
import numpy as np
import matplotlib

# Correct path
# path = r'C:\\Users\\IT2.student4\\Desktop\\computer vision swssion 1\\'

# Load images
img1 = cv2.imread('F:\\University\\Fourth year\\Computer Vision\\computer vision lecture 1\\P1.jfif')
img2 = cv2.imread('F:\\University\\Fourth year\\Computer Vision\\computer vision lecture 1\\P2.jfif')

# Check if images are loaded
if img1 is None or img2 is None:
    print("Error: One or both images could not be loaded.")
    exit()

# Print shapes
print("Image 1 shape: ", img1.shape)
print("Image 2 shape: ", img2.shape)

r1, c1, _ = img1.shape
r2, c2, _ = img2.shape

# Show images
cv2.imshow('image 1', img1)
cv2.waitKey(2000)
cv2.destroyAllWindows()

cv2.imshow('image 2', img2)
cv2.waitKey(2000)
cv2.destroyAllWindows()

# Show single channel
cv2.imshow('image 1 - Channel 0', img1[:, :, 0])
cv2.waitKey(2000)
cv2.destroyAllWindows()

# Convert and display RGB
P1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
cv2.imshow('image 1 - RGB', P1_rgb)
cv2.waitKey(2000)
cv2.destroyAllWindows()

# Convert second image to grayscale and display
P2_g = cv2.imread('P2.jfif', cv2.IMREAD_GRAYSCALE)
cv2.imshow('image 2 - Grayscale', P2_g)
cv2.waitKey(2000)
cv2.destroyAllWindows()

# Save images
name1 = '/P1_rgb_w.jpeg'
name2 = '/P2_g_w.jpeg'

cv2.imwrite(f'{path}{name1}', P1_rgb)
cv2.imwrite(f'{path}{name2}', P2_g)

# Print specific pixel
print(img1[50, 75, :])

# Modify image
img1_c = img1.copy()
img1_c[0:15, :, :] = 0
cv2.imshow('img1_c', img1_c)
cv2.waitKey(2000)
cv2.destroyAllWindows()

# Add rectangle to img2
rec2 = img2.copy()
rec2[round(r2 / 2) - 7:round(r2 / 2) + 8, round(c2 / 2) - 7:round(c2 / 2) + 8, :] = 255
cv2.imshow('rec2', rec2)
cv2.waitKey(2000)
cv2.destroyAllWindows()

# Modify image to show only blue
img11 = img1.copy()
img11[:, :, 2] = 0
img11[:, :, 1] = 0
cv2.imshow('colorful blue', img11)
cv2.waitKey(2000)
cv2.destroyAllWindows()
