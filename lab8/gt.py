import cv2
import numpy as np
import matplotlib.pyplot as plt
# 1. Read the colorful image (I)
def generate_gaussian_kernel(sig, siz):
    result = np.zeros((siz, siz))
    x = [i for i in range((-siz // 2) + 1, (siz // 2) + 1, 1)]
    y = x
    for i in range(siz):
        for j in range(siz):
            a = 1 / (2 * np.pi * (sig ** 2))
            result[i][j] = a * np.exp(-((x[i] ** 2) + (y[j] ** 2)) / (2 * (sig ** 2)))
    return result
def convolution(img, mask, size):
    r= img.shape[0]
    c= img.shape[1]
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
def read_image(path):
    return cv2.imread(path)

# 2. Create resized versions of the image
sizes = [(400, 400), (200, 200), (100, 100), (50, 50)]
def resize_images(image, sizes):
    return [cv2.resize(image, size) for size in sizes]

# 3. Convert resized images to grayscale
def convert_to_grayscale(images):
    return [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

# 4. Apply Gaussian filters to calculate octaves
def gaussian_filter(image, sigma):
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2==0:
        kernel_size=kernel_size+1
    return convolution(image,generate_gaussian_kernel(sigma, kernel_size),kernel_size)

def calculate_octaves(grayscale_images, sigma_initial=2, k=np.sqrt(2)):
    octaves = []
    for gray_img in grayscale_images:
        octave = []
        sigma = sigma_initial
        for _ in range(4):
            octave.append(gaussian_filter(gray_img, sigma))
            sigma *= k
        octaves.append(octave)
    return octaves

# 5. Calculate Difference of Gaussians (DOGs)
def calculate_dogs(octaves):
    dogs = []
    for octave in octaves:
        dog = [octave[i + 1] - octave[i] for i in range(len(octave) - 1)]
        dogs.append(dog)
    return dogs

# 6. Find local extrema in each DOG

def find_local_extrema(dogs):
    extrema_images = []
    for dog_set in dogs:
        height, width = dog_set[1].shape
        extrema = np.zeros((height, width), dtype=np.float32)
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                cube = np.stack(
                    [dog[i - 1:i + 2, j - 1:j + 2] for dog in dog_set], axis=0
                )
                extrema[i, j] = np.max(cube) if cube[1, 1, 1] == np.max(cube) else (
                    np.min(cube) if cube[1, 1, 1] == np.min(cube) else 0
                )
        extrema_images.append(extrema)
    return extrema_images

# 7. Multiply extremes by corresponding DOG images

def multiply_extremes(extrema_images, dogs):
    return [extrema * dog_set[1] for extrema, dog_set in zip(extrema_images, dogs)]

# 8. Take absolute values
def take_absolute(images):
    return [np.abs(img) for img in images]

# 9. Apply Thresholding
def apply_threshold(images, threshold):
    return [(img > threshold).astype(np.uint8) for img in images]

# 10. Find key-points and rescale coordinates
def find_keypoints(images, scales):
    keypoints = []
    for img, scale in zip(images, scales):
        points = np.argwhere(img > 0)
        keypoints.append([(int(x * scale), int(y * scale)) for y, x in points])
    return keypoints

def draw_keypoints(image, keypoints, radii):
    result = image.copy()
    for i, points in enumerate(keypoints):
        for point in points:
            cv2.circle(result, point, radii[i], (255, 0, 0), 1)
    return result

# Main pipeline
def main(image_path):
    # Read image
    image = read_image(image_path)

    # Resized images
    resized_images = resize_images(image, sizes)

    # Convert to grayscale
    grayscale_images = convert_to_grayscale(resized_images)

    # Calculate octaves
    octaves = calculate_octaves(grayscale_images)

    # Calculate DOGs
    dogs = calculate_dogs(octaves)

    # Find local extrema
    extrema_images = find_local_extrema(dogs)

    # Multiply extremes by DOGs
    multiplied_images = multiply_extremes(extrema_images, dogs)

    # Absolute values
    absolute_images = take_absolute(multiplied_images)

    # Thresholding
    thresholded_images = apply_threshold(absolute_images, threshold=0.03)

    # Find and draw keypoints
    scales = [1, 2, 4, 8]
    keypoints = find_keypoints(thresholded_images, scales)
    final_image = draw_keypoints(resized_images[0], keypoints, radii=[5, 10, 15, 20])

    # Show final image
    plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

main("I0.jpg")
