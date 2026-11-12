import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths to images
path1 = "D:\\Computer Vision\\lab 2\\P1.jfif"
path2 = "D:\\Computer Vision\\lab 2\\P2.jfif"

# Load P1 as a grayscale image
P1_g = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)

# Ensure image is loaded
if P1_g is None:
    print("Error: Image P1 could not be loaded.")
    exit()

# Get image dimensions
r, c = P1_g.shape
print(f"Image dimensions: Rows={r}, Columns={c}")

# Initialize required vectors and arrays
h = np.zeros(256, dtype=np.int32)          # Histogram
idx = np.arange(256)                       # Intensity levels
h_sum = np.zeros(256, dtype=np.int32)      # Cumulative histogram
equalized_img = np.zeros_like(P1_g, dtype=np.uint8)  # Equalized image

# Record the frequency of each gray level
for pixel in P1_g.flatten():
    h[pixel] += 1

# Plot original histogram
plt.figure()
plt.bar(idx, h, color='blue', alpha=0.7)
plt.title("Original Histogram (h)")
plt.xlabel("Gray Level")
plt.ylabel("Frequency")
plt.show()

# Compute the cumulative histogram
h_sum[0] = h[0]
for i in range(1, len(h)):
    h_sum[i] = h_sum[i - 1] + h[i]

# Plot cumulative histogram
plt.figure()
plt.bar(idx, h_sum, color='green', alpha=0.7)
plt.title("Cumulative Histogram (h_sum)")
plt.xlabel("Gray Level")
plt.ylabel("Cumulative Frequency")
plt.show()

# Normalize the cumulative histogram
norm_sum = (h_sum * 255) // h_sum[-1]  # Scaling to max gray level (255)

# Plot normalized cumulative histogram
plt.figure()
plt.bar(idx, norm_sum, color='orange', alpha=0.7)
plt.title("Normalized Cumulative Histogram (norm_sum)")
plt.xlabel("Gray Level")
plt.ylabel("Normalized Frequency")
plt.show()

# Replace gray levels in the original image with equalized levels
for i in range(r):
    for j in range(c):
        equalized_img[i, j] = norm_sum[P1_g[i, j]]

# Calculate histogram of equalized image
h_eq = np.zeros(256, dtype=np.int32)
for pixel in equalized_img.flatten():
    h_eq[pixel] += 1

# Display results in one figure
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.bar(idx, h, color='blue', alpha=0.7)
plt.title("Original Histogram")

plt.subplot(2, 2, 2)
plt.bar(idx, h_eq, color='red', alpha=0.7)
plt.title("Equalized Histogram")

plt.subplot(2, 2, 3)
plt.imshow(P1_g, cmap='gray')
plt.title("Original Image")

plt.subplot(2, 2, 4)
plt.imshow(equalized_img, cmap='gray')
plt.title("Equalized Image")

plt.tight_layout()
plt.show()

# # Display images using OpenCV
# cv2.imshow('Original Image', P1_g)
# cv2.imshow('Equalized Image', equalized_img)
# cv2.waitKey(5000)
# cv2.destroyAllWindows()
