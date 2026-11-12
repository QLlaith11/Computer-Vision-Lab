import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the colorful image
def read_image(image_path):
    """Read the input image"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

# Step 2: Resize image to different scales
def create_image_pyramid(img):
    """Create 4 different sized versions of the image"""
    sizes = [400, 200, 100, 50]
    resized_images = []
    
    # First resize to 400x400 as base
    base_img = cv2.resize(img, (400, 400))
    resized_images.append(base_img)
    
    # Then create other sizes
    for size in sizes[1:]:
        resized = cv2.resize(img, (size, size))
        resized_images.append(resized)
    
    return resized_images, base_img

# Step 3: Convert to grayscale
def convert_to_grayscale(images):
    """Convert all images to grayscale"""
    gray_images = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_images.append(gray)
    
    return gray_images

# Step 4: Calculate octave with different Gaussian filters
def calculate_octave(gray_img):
    """Apply 4 Gaussian filters with different sigma values"""
    K = np.sqrt(2)
    sigma1 = 2
    sigmas = [sigma1 * (K ** i) for i in range(4)]
    
    gaussian_images = []
    for sigma in sigmas:
        ksize = int(6 * sigma)
        if ksize % 2 == 0:  # Ensure odd kernel size
            ksize += 1
        
        # Apply Gaussian filter
        gaussian = cv2.GaussianBlur(gray_img, (ksize, ksize), sigma)
        gaussian_images.append(gaussian)
    
    return gaussian_images, sigmas

# Step 5: Calculate Difference of Gaussians (DOG)
def calculate_dog(gaussian_images):
    """Calculate DOG by subtracting consecutive Gaussian images"""
    dogs = []
    for i in range(len(gaussian_images) - 1):
        dog = gaussian_images[i+1].astype(np.float32) - gaussian_images[i].astype(np.float32)
        dogs.append(dog)
    
    return dogs

# Step 6: Find local extrema
def find_local_extrema(dogs):
    """Find local maxima and minima using 3x3 window across scales"""
    if len(dogs) < 3:
        return None
    
    # Use middle DOG image
    mid = dogs[1]
    upper = dogs[0]
    lower = dogs[2]
    
    extrema = np.zeros_like(mid)
    
    # Iterate through middle image (excluding borders)
    for i in range(1, mid.shape[0] - 1):
        for j in range(1, mid.shape[1] - 1):
            # Get 3x3 neighborhoods
            mid_region = mid[i-1:i+2, j-1:j+2]
            upper_region = upper[i-1:i+2, j-1:j+2]
            lower_region = lower[i-1:i+2, j-1:j+2]
            
            center_val = mid[i, j]
            
            # Stack all 27 values (3x3x3)
            all_vals = np.concatenate([mid_region.flatten(), 
                                      upper_region.flatten(), 
                                      lower_region.flatten()])
            
            # Check if center is maximum or minimum
            if center_val == np.max(all_vals) or center_val == np.min(all_vals):
                extrema[i, j] = 1
    
    return extrema

# Step 7: Multiply extrema by corresponding DOG
def multiply_extrema_dog(extrema, dog):
    """Multiply extrema mask by DOG image"""
    if extrema is None:
        return None
    return extrema * dog

# Step 8: Take absolute values
def take_absolute(extrema_dog):
    """Take absolute values of extrema"""
    if extrema_dog is None:
        return None
    return np.abs(extrema_dog)

# Step 9: Apply thresholding
def apply_threshold(extrema_abs, threshold=10):
    """Apply thresholding to filter weak extrema"""
    if extrema_abs is None:
        return None
    extrema_thresh = np.where(extrema_abs > threshold, extrema_abs, 0)
    return extrema_thresh

# Step 10: Find keypoint indices and rescale
def find_and_rescale_keypoints(extrema_images, sizes, base_size=400):
    """Find keypoints and rescale coordinates"""
    all_keypoints = []
    scale_factors = [base_size / size for size in sizes]
    
    for idx, (extrema, scale) in enumerate(zip(extrema_images, scale_factors)):
        if extrema is None:
            all_keypoints.append([])
            continue
        
        # Find non-zero points
        y_coords, x_coords = np.nonzero(extrema)
        
        # Rescale coordinates
        keypoints = []
        for y, x in zip(y_coords, x_coords):
            scaled_x = int(x * scale)
            scaled_y = int(y * scale)
            keypoints.append((scaled_x, scaled_y))
        
        all_keypoints.append(keypoints)
    
    return all_keypoints, scale_factors

# Visualization function
def draw_keypoints(img, all_keypoints, scale_factors):
    """Draw circles around keypoints on the base 400x400 image"""
    img_with_keypoints = img.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    radii = [3, 5, 7, 9]
    
    for idx, keypoints in enumerate(all_keypoints):
        for (x, y) in keypoints:
            # Make sure coordinates are within bounds
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                cv2.circle(img_with_keypoints, (x, y), radii[idx], colors[idx], 2)
    
    return img_with_keypoints

# Main pipeline
def sift_dog_pipeline(image_path, threshold=10):
    """Complete SIFT DOG pipeline"""
    print("Step 1: Reading image...")
    img = read_image(image_path)
    
    print("Step 2: Creating image pyramid...")
    resized_images, base_img = create_image_pyramid(img)
    sizes = [400, 200, 100, 50]
    
    print("Step 3: Converting to grayscale...")
    gray_images = convert_to_grayscale(resized_images)
    
    # Process each octave
    all_extrema = []
    all_dogs = []
    
    for i, gray_img in enumerate(gray_images):
        print(f"\nProcessing octave {i+1} (size: {sizes[i]}x{sizes[i]})...")
        
        print(f"  Step 4: Calculating Gaussian octave...")
        gaussian_images, sigmas = calculate_octave(gray_img)
        
        print(f"  Step 5: Calculating DOG...")
        dogs = calculate_dog(gaussian_images)
        all_dogs.append(dogs)
        
        print(f"  Step 6: Finding local extrema...")
        extrema = find_local_extrema(dogs)
        
        print(f"  Step 7: Multiplying extrema by DOG...")
        extrema_dog = multiply_extrema_dog(extrema, dogs[1])
        
        print(f"  Step 8: Taking absolute values...")
        extrema_abs = take_absolute(extrema_dog)
        
        print(f"  Step 9: Applying threshold...")
        extrema_thresh = apply_threshold(extrema_abs, threshold)
        
        all_extrema.append(extrema_thresh)
    
    print("\nStep 10: Finding and rescaling keypoints...")
    all_keypoints, scale_factors = find_and_rescale_keypoints(all_extrema, sizes)
    
    # Count keypoints
    for i, kps in enumerate(all_keypoints):
        print(f"  Octave {i+1}: {len(kps)} keypoints detected")
    
    # Draw keypoints on the base 400x400 image
    img_with_keypoints = draw_keypoints(base_img, all_keypoints, scale_factors)
    
    return img_with_keypoints, all_keypoints, all_dogs, all_extrema, gray_images

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "I0.jpg"
    
    # Run pipeline
    result_img, keypoints, dogs, extrema, gray_images = sift_dog_pipeline(image_path, threshold=10)
    
    plt.imshow(result_img)
    plt.title('Original with Keypoints')
    plt.axis('off')
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    # Original with keypoints
    plt.subplot(2, 3, 1)
    plt.imshow(result_img)
    plt.title('Original with Keypoints')
    plt.axis('off')
    
    # Show some DOG images
    for i in range(min(4, len(dogs))):
        plt.subplot(2, 3, i+2)
        if dogs[i]:
            plt.imshow(dogs[i][1], cmap='gray')
            plt.title(f'DOG Octave {i+1}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()
