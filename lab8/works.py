import cv2
import numpy as np

# ==================== PARAMETERS ====================
SIZES = [400, 200, 100, 50]
SCALE_FACTORS = [1, 2, 4, 8]
THRESHOLDS = [50, 10, 0, 0]  # th1, th2, th3, th4
K = np.sqrt(2)
SIGMAS = [2, 2*K, 2*(K**2), 2*(K**3)]
MATCH_THRESHOLD = 17


# ==================== DOG FEATURE EXTRACTION ====================
def create_octave(gray_img):
    """Create octave using 4 Gaussian filters"""
    octave = []
    for sigma in SIGMAS:
        ksize = int(6 * sigma)
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(gray_img, (ksize, ksize), sigma)
        octave.append(blurred)
    return octave


def calculate_dog(octave):
    """Calculate Difference of Gaussians"""
    dogs = []
    for i in range(len(octave) - 1):
        dog = octave[i + 1] - octave[i]
        dogs.append(dog)
    return dogs


def find_local_extremes(dogs):
    """Find local maxima and minima in middle DOG using 3x3 window"""
    h, w = dogs[1].shape
    extremes = np.zeros((h, w))
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            center = dogs[1][i, j]
            
            # Get all 26 neighbors from 3 DOG levels
            neighbors = []
            for d in range(3):
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if not (d == 1 and di == 0 and dj == 0):
                            neighbors.append(dogs[d][i + di, j + dj])
            
            neighbors = np.array(neighbors)
            
            # Check if center is maximum or minimum
            if center > np.max(neighbors) or center < np.min(neighbors):
                extremes[i, j] = center
    
    return extremes


# ==================== HOG FEATURE DESCRIPTION ====================
def calculate_gradients(gray_img):
    """Calculate gradients Ix, Iy, magnitude and direction"""
    Ix = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(Ix**2 + Iy**2)
    direction = np.arctan2(Iy, Ix)  # Di = arctan2(Iy, Ix)
    return magnitude, direction


def compute_descriptor(magnitude, direction, y, x):
    """Compute 128-element HOG descriptor for a keypoint"""
    h, w = magnitude.shape
    
    # Check boundaries for 17x17 window
    if y < 8 or y >= h - 8 or x < 8 or x >= w - 8:
        return None
    
    # Step 1: Take 17x17 window (w1) around the keypoint
    mag_w1 = magnitude[y - 8:y + 9, x - 8:x + 9].copy()
    dir_w1 = direction[y - 8:y + 9, x - 8:x + 9].copy()
    
    # Step 2: Calculate dominant orientation (OD)
    bins = np.linspace(-np.pi, np.pi, 9)  # 8 bins for 8 directions
    hist, _ = np.histogram(dir_w1.flatten(), bins=bins, weights=mag_w1.flatten())
    dominant_bin = np.argmax(hist)
    OD = (bins[dominant_bin] + bins[dominant_bin + 1]) / 2
    
    # Step 3: Re-rotate elements by (-OD) using transformation matrix
    center = (8, 8)
    angle_degrees = np.degrees(-OD)
    rot_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    mag_rotated = cv2.warpAffine(mag_w1, rot_matrix, (17, 17))
    dir_rotated = cv2.warpAffine(dir_w1, rot_matrix, (17, 17))
    dir_rotated = dir_rotated - OD  # Adjust directions
    
    # Step 4: Take 16x16 sub-window (w2) from left upper corner
    mag_w2 = mag_rotated[0:16, 0:16]
    dir_w2 = dir_rotated[0:16, 0:16]
    
    # Step 5: Calculate histogram of gradients (16 histograms x 8 bins = 128)
    descriptor = []
    for i in range(4):
        for j in range(4):
            # 4x4 sub-window
            mag_sub = mag_w2[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4]
            dir_sub = dir_w2[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4]
            # 8-bin histogram
            h_sub, _ = np.histogram(dir_sub.flatten(), bins=bins, 
                                    weights=mag_sub.flatten())
            descriptor.extend(h_sub)
    
    descriptor = np.array(descriptor, dtype=np.float64)
    
    # Normalize descriptor
    norm = np.linalg.norm(descriptor)
    if norm > 0:
        descriptor = descriptor / norm
    
    return descriptor


# ==================== MAIN FEATURE EXTRACTION ====================
def extract_features(img):
    """Extract keypoints and compute descriptors for an image"""
    all_keypoints = []
    all_descriptors = []
    
    for idx in range(4):
        size = SIZES[idx]
        scale = SCALE_FACTORS[idx]
        th = THRESHOLDS[idx]
        
        # Resize image
        resized = cv2.resize(img, (size, size))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float64)
        
        # Create octave and calculate DOG
        octave = create_octave(gray)
        dogs = calculate_dog(octave)
        
        # Find local extremes
        extremes = find_local_extremes(dogs)
        
        # Multiply by DOG and take absolute
        extremes = np.abs(extremes * dogs[1])
        
        # Apply threshold
        extremes[extremes < th] = 0
        
        # Get keypoint coordinates
        ky, kx = np.where(extremes > 0)
        
        # Calculate gradients
        magnitude, direction = calculate_gradients(gray)
        
        # Calculate descriptors for each keypoint
        for y, x in zip(ky, kx):
            desc = compute_descriptor(magnitude, direction, y, x)
            if desc is not None:
                # Rescale coordinates
                all_keypoints.append((y * scale, x * scale))
                all_descriptors.append(desc)
    
    return all_keypoints, all_descriptors


# ==================== FEATURE MATCHING ====================
def euclidean_distance(desc1, desc2):
    """Calculate Euclidean distance between two descriptors"""
    diff = desc1 - desc2
    return np.sqrt(np.sum(diff ** 2))


def match_features(kp1, desc1, kp2, desc2, threshold):
    """Match features using Euclidean distance"""
    matches = []
    
    for i, d1 in enumerate(desc1):
        best_dist = float('inf')
        best_j = -1
        
        for j, d2 in enumerate(desc2):
            dist = euclidean_distance(d1, d2)
            if dist < best_dist:
                best_dist = dist
                best_j = j
        
        if best_dist < threshold:
            matches.append((i, best_j, best_dist))
    
    return matches


# ==================== VISUALIZATION ====================
def draw_keypoints(img, keypoints, color=(0, 255, 0)):
    """Draw circles around keypoints"""
    img_copy = img.copy()
    h, w = img.shape[:2]
    
    for y, x in keypoints:
        # Scale to image size
        x_scaled = int(x * w / 400)
        y_scaled = int(y * h / 400)
        cv2.circle(img_copy, (x_scaled, y_scaled), 5, color, 2)
    
    return img_copy


def visualize_matches(img1, img2, kp1, kp2, matches):
    """Visualize matched keypoints between two images"""
    # Resize for display
    img1_vis = cv2.resize(img1, (400, 400))
    img2_vis = cv2.resize(img2, (400, 400))
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    result = np.hstack([img1_vis, img2_vis])
    
    for i, j, dist in matches:
        y1, x1 = kp1[i]
        y2, x2 = kp2[j]
        
        # Scale to visualization size
        x1_vis = int(x1 * 400 / w1)
        y1_vis = int(y1 * 400 / h1)
        x2_vis = int(x2 * 400 / w2) + 400
        y2_vis = int(y2 * 400 / h2)
        
        # Draw keypoints
        cv2.circle(result, (x1_vis, y1_vis), 4, (0, 255, 0), 2)
        cv2.circle(result, (x2_vis, y2_vis), 4, (0, 255, 0), 2)
        # Draw matching lines
        cv2.line(result, (x1_vis, y1_vis), (x2_vis, y2_vis), (255, 0, 0), 1)
    
    return result


# ==================== MAIN FUNCTION ====================
def main():
    # Read colorful image I8
    I8 = cv2.imread('I8.png')  # Change to your image path
    
    if I8 is None:
        print("Error: Could not load image!")
        return
    
    # Create I9 by rotating I8 by 45 degrees
    h, w = I8.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    I9 = cv2.warpAffine(I8, rotation_matrix, (w, h))
    
    print("Extracting features from I8...")
    kp1, desc1 = extract_features(I8)
    print(f"Found {len(kp1)} keypoints in I8")
    
    print("Extracting features from I9...")
    kp2, desc2 = extract_features(I9)
    print(f"Found {len(kp2)} keypoints in I9")
    
    # Match keypoints
    print("Matching features...")
    matches = match_features(kp1, desc1, kp2, desc2, MATCH_THRESHOLD)
    print(f"Found {len(matches)} matches")
    
    # Visualize results
    result = visualize_matches(I8, I9, kp1, kp2, matches)
    
    # Display images
    cv2.imshow('Original Image (I8)', cv2.resize(I8, (400, 400)))
    cv2.imshow('Rotated Image (I9)', cv2.resize(I9, (400, 400)))
    cv2.imshow('SIFT Matches', result)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save result
    cv2.imwrite('sift_matches_result.jpg', result)
    print("Result saved to 'sift_matches_result.jpg'")


if __name__ == "__main__":
    main()
