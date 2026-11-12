import cv2
import numpy as np

# ============================================================================
# DOG (Difference of Gaussians) - Feature Extraction
# ============================================================================

def create_gaussian_kernel(sigma):
    """Create Gaussian kernel with size [6*sigma, 6*sigma]"""
    size = int(6 * sigma)
    if size % 2 == 0:
        size += 1
    kernel = cv2.getGaussianKernel(size, sigma)
    kernel = kernel * kernel.T
    return kernel

def compute_dog_octave(gray_image):
    """Compute DOG for one octave with 4 different sigma values"""
    K = np.sqrt(2)
    sigma1 = 2
    sigmas = [sigma1 * (K ** i) for i in range(4)]
    
    # Apply Gaussian filters
    gaussian_images = []
    for sigma in sigmas:
        kernel = create_gaussian_kernel(sigma)
        blurred = cv2.filter2D(gray_image, -1, kernel)
        gaussian_images.append(blurred)
    
    # Calculate DOGs
    dogs = []
    for i in range(3):
        dog = gaussian_images[i+1] - gaussian_images[i]
        dogs.append(dog)
    
    return dogs, gaussian_images

def find_local_extrema(dogs):
    """Find local extrema (maxima and minima) in DOG octave"""
    extrema = np.zeros_like(dogs[1], dtype=np.uint8)
    
    h, w = dogs[1].shape
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            # Get 3x3 windows from all three levels
            lower = dogs[0][y-1:y+2, x-1:x+2]
            mid = dogs[1][y-1:y+2, x-1:x+2]
            upper = dogs[2][y-1:y+2, x-1:x+2]
            
            center_val = mid[1, 1]
            
            # Check if it's a maximum
            if (center_val > np.max(lower) and 
                center_val > np.max(upper) and
                center_val >= np.max(mid)):
                extrema[y, x] = 1
            
            # Check if it's a minimum
            elif (center_val < np.min(lower) and 
                  center_val < np.min(upper) and
                  center_val <= np.min(mid)):
                extrema[y, x] = 1
    
    return extrema

def extract_features_dog(image, th1=50, th2=10, th3=0, th4=0):
    """Extract features using DOG for all octaves"""
    # Resize to different scales
    sizes = [400, 200, 100, 50]
    scale_factors = [1, 2, 4, 8]
    thresholds = [th1, th2, th3, th4]
    
    all_keypoints = []
    all_descriptors_data = []  # Store gradient info for descriptor calculation
    
    for idx, (size, scale, th) in enumerate(zip(sizes, scale_factors, thresholds)):
        # Resize image
        resized = cv2.resize(image, (size, size))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Compute DOG octave
        dogs, gaussians = compute_dog_octave(gray)
        
        # Find extrema
        extrema = find_local_extrema(dogs)
        
        # Multiply by mid-level DOG and take absolute
        extrema_weighted = np.abs(extrema.astype(np.float32) * dogs[1])
        
        # Apply threshold
        extrema_weighted[extrema_weighted < th] = 0
        
        # Find keypoint coordinates
        keypoint_coords = np.argwhere(extrema_weighted > 0)
        
        # Calculate gradients for descriptor
        Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        D = np.arctan2(Iy, Ix)
        
        # Rescale coordinates and store
        for coord in keypoint_coords:
            y, x = coord
            scaled_y = int(y * scale)
            scaled_x = int(x * scale)
            all_keypoints.append((scaled_x, scaled_y, size, idx))
            all_descriptors_data.append((x, y, Ix, Iy, D, gray))
    
    return all_keypoints, all_descriptors_data

# ============================================================================
# HOG (Histogram of Oriented Gradients) - Feature Description
# ============================================================================

def calculate_dominant_orientation(D, x, y, window_size=17):
    """Calculate dominant orientation in a window around keypoint"""
    half = window_size // 2
    h, w = D.shape
    
    y1 = max(0, y - half)
    y2 = min(h, y + half + 1)
    x1 = max(0, x - half)
    x2 = min(w, x + half + 1)
    
    window = D[y1:y2, x1:x2]
    
    # Create histogram of orientations (8 bins)
    hist, _ = np.histogram(window.flatten(), bins=8, range=(-np.pi, np.pi))
    dominant_bin = np.argmax(hist)
    dominant_orientation = -np.pi + (dominant_bin + 0.5) * (2 * np.pi / 8)
    
    return dominant_orientation

def rotate_window(Ix, Iy, D, x, y, angle, window_size=17):
    """Rotate window around keypoint"""
    half = window_size // 2
    h, w = D.shape
    
    # Create rotation matrix
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    
    rotated_Ix = np.zeros((window_size, window_size), dtype=np.float32)
    rotated_Iy = np.zeros((window_size, window_size), dtype=np.float32)
    rotated_D = np.zeros((window_size, window_size), dtype=np.float32)
    
    for i in range(window_size):
        for j in range(window_size):
            # Offset from center
            dy = i - half
            dx = j - half
            
            # Rotate coordinates
            new_x = int(x + dx * cos_a - dy * sin_a)
            new_y = int(y + dx * sin_a + dy * cos_a)
            
            # Check bounds
            if 0 <= new_y < h and 0 <= new_x < w:
                rotated_Ix[i, j] = Ix[new_y, new_x]
                rotated_Iy[i, j] = Iy[new_y, new_x]
                rotated_D[i, j] = D[new_y, new_x]
    
    return rotated_Ix, rotated_Iy, rotated_D

def compute_hog_descriptor(rotated_D, rotated_Ix, rotated_Iy):
    """Compute 128-dimensional HOG descriptor"""
    # Take 16x16 window from top-left of 17x17 rotated window
    D_16 = rotated_D[:16, :16]
    Ix_16 = rotated_Ix[:16, :16]
    Iy_16 = rotated_Iy[:16, :16]
    
    # Calculate magnitude
    magnitude = np.sqrt(Ix_16**2 + Iy_16**2)
    
    descriptor = []
    
    # Divide into 4x4 cells (16 cells total)
    for i in range(4):
        for j in range(4):
            cell_D = D_16[i*4:(i+1)*4, j*4:(j+1)*4]
            cell_mag = magnitude[i*4:(i+1)*4, j*4:(j+1)*4]
            
            # Compute histogram (8 bins)
            hist, _ = np.histogram(cell_D.flatten(), bins=8, range=(-np.pi, np.pi), 
                                  weights=cell_mag.flatten())
            descriptor.extend(hist)
    
    # Normalize descriptor
    descriptor = np.array(descriptor)
    norm = np.linalg.norm(descriptor)
    if norm > 0:
        descriptor = descriptor / norm
    
    return descriptor

def compute_descriptors(keypoints, descriptors_data):
    """Compute descriptors for all keypoints"""
    descriptors = []
    valid_keypoints = []
    
    for kp, data in zip(keypoints, descriptors_data):
        x, y, Ix, Iy, D, gray = data
        h, w = D.shape
        
        # Check if window fits
        if x < 9 or y < 9 or x >= w - 9 or y >= h - 9:
            continue
        
        # Calculate dominant orientation
        OD = calculate_dominant_orientation(D, x, y, 17)
        
        # Rotate window
        rot_Ix, rot_Iy, rot_D = rotate_window(Ix, Iy, D, x, y, OD, 17)
        
        # Compute descriptor
        descriptor = compute_hog_descriptor(rot_D, rot_Ix, rot_Iy)
        
        descriptors.append(descriptor)
        valid_keypoints.append(kp)
    
    return valid_keypoints, np.array(descriptors)

# ============================================================================
# Matching
# ============================================================================

def match_features(desc1, desc2, kp1, kp2, threshold=17):
    """Match features using Euclidean distance"""
    matches = []
    
    for i, d1 in enumerate(desc1):
        min_dist = float('inf')
        min_idx = -1
        
        for j, d2 in enumerate(desc2):
            dist = np.sqrt(np.sum((d1 - d2) ** 2))
            if dist < min_dist:
                min_dist = dist
                min_idx = j
        
        if min_dist < threshold:
            matches.append((i, min_idx, min_dist))
    
    return matches

def draw_matches(img1, img2, kp1, kp2, matches):
    """Draw matching results"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create output image
    output = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    output[:h1, :w1] = img1
    output[:h2, w1:w1+w2] = img2
    
    # Draw matches
    for idx1, idx2, dist in matches:
        x1, y1, _, _ = kp1[idx1]
        x2, y2, _, _ = kp2[idx2]
        
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2 + w1), int(y2))
        
        color = (0, 255, 0)
        cv2.line(output, pt1, pt2, color, 1)
        cv2.circle(output, pt1, 3, (255, 0, 0), -1)
        cv2.circle(output, pt2, 3, (0, 0, 255), -1)
    
    return output

# ============================================================================
# Main Program
# ============================================================================

# Read images
I8 = cv2.imread('I8.png')
I9 = cv2.imread('I9.png')

if I8 is None or I9 is None:
    print("Error: Could not read images I8.jpg or I9.jpg")
    exit()

print("Extracting features from I8...")
keypoints1, desc_data1 = extract_features_dog(I8, th1=30, th2=10, th3=0, th4=0)
print(f"Found {len(keypoints1)} keypoints in I8")

print("Computing descriptors for I8...")
keypoints1, descriptors1 = compute_descriptors(keypoints1, desc_data1)
print(f"Computed {len(descriptors1)} descriptors for I8")

print("Extracting features from I9...")
keypoints2, desc_data2 = extract_features_dog(I9, th1=50, th2=10, th3=0, th4=0)
print(f"Found {len(keypoints2)} keypoints in I9")

print("Computing descriptors for I9...")
keypoints2, descriptors2 = compute_descriptors(keypoints2, desc_data2)
print(f"Computed {len(descriptors2)} descriptors for I9")

print("Matching features...")
matches = match_features(descriptors1, descriptors2, keypoints1, keypoints2, threshold=17)
print(f"Found {len(matches)} matches")

# Draw results
result = draw_matches(I8, I9, keypoints1, keypoints2, matches)

cv2.imshow('SIFT Matches', result)
cv2.imwrite('sift_matches.jpg', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Done! Results saved as 'sift_matches.jpg'")
