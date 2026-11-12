import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_gaussian_kernel(sigma):
    size = int(6 * sigma)
    if size % 2 == 0:
        size += 1
    kernel = np.zeros((size, size))
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    return kernel / np.sum(kernel)


def apply_gaussian_filter(image, sigma):
    kernel = create_gaussian_kernel(sigma)
    return cv2.filter2D(image, -1, kernel)


def create_octave(gray_image, K=np.sqrt(2), sigma1=2):
    octave = []
    sigma = sigma1
    
    for _ in range(4):
        filtered = apply_gaussian_filter(gray_image, sigma)
        octave.append(filtered)
        sigma = K * sigma
    
    return octave


def calculate_dog(octave):
    return [octave[i+1] - octave[i] for i in range(len(octave) - 1)]


def find_local_extrema(dogs):
    mid_level = dogs[1]
    extrema = np.zeros_like(mid_level)
    h, w = mid_level.shape
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            window_lower = dogs[0][i-1:i+2, j-1:j+2]
            window_mid = dogs[1][i-1:i+2, j-1:j+2]
            window_upper = dogs[2][i-1:i+2, j-1:j+2]
            
            center_val = mid_level[i, j]
            all_vals = np.concatenate([window_lower.flatten(), window_mid.flatten(), window_upper.flatten()])
            
            center_idx = len(window_lower.flatten()) + 4
            all_vals_no_center = np.concatenate([all_vals[:center_idx], all_vals[center_idx+1:]])
            
            if center_val > np.max(all_vals_no_center) or center_val < np.min(all_vals_no_center):
                extrema[i, j] = center_val
    
    return extrema


def extract_keypoints_single_scale(gray_image, scale_factor, th1=50):
    octave = create_octave(gray_image)
    dogs = calculate_dog(octave)
    extrema = find_local_extrema(dogs)
    extrema_abs = np.abs(extrema * dogs[1])
    extrema_thresholded = np.where(extrema_abs < th1, 0, extrema_abs)
    
    y_coords, x_coords = np.where(extrema_thresholded > 0)
    keypoints = [(int(x * scale_factor), int(y * scale_factor), x, y) for y, x in zip(y_coords, x_coords)]
    
    return keypoints, gray_image


def extract_keypoints(image_color, th1=50):
    sizes = [400, 200, 100, 50]
    scale_factors = [1, 2, 4, 8]
    all_keypoints = []
    
    for size, scale in zip(sizes, scale_factors):
        resized = cv2.resize(image_color, (size, size))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
        keypoints, gray_img = extract_keypoints_single_scale(gray, scale, th1)
        all_keypoints.extend([(kp[0], kp[1], scale, kp[2], kp[3], gray_img) for kp in keypoints])
    
    return all_keypoints


def calculate_gradients(gray_image):
    Ix = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    D = np.arctan2(Iy, Ix)
    magnitude = np.sqrt(Ix**2 + Iy**2)
    return Ix, Iy, D, magnitude


def calculate_dominant_orientation(D_window, mag_window):
    hist = np.zeros(36)
    h, w = D_window.shape
    
    for i in range(h):
        for j in range(w):
            angle_deg = (np.degrees(D_window[i, j]) + 360) % 360
            bin_idx = int(angle_deg / 10) % 36
            hist[bin_idx] += mag_window[i, j]
    
    return np.radians(np.argmax(hist) * 10)


def calculate_descriptor(gray_image, x, y, D, magnitude):
    h, w = gray_image.shape
    
    if x < 8 or y < 8 or x >= w - 9 or y >= h - 9:
        return None
    
    D_window = D[y-8:y+9, x-8:x+9]
    mag_window = magnitude[y-8:y+9, x-8:x+9]
    
    if D_window.shape != (17, 17):
        return None
    
    O_D = calculate_dominant_orientation(D_window, mag_window)
    D_rotated = D_window - O_D
    D_sub = D_rotated[0:16, 0:16]
    mag_sub = mag_window[0:16, 0:16]
    
    descriptor = []
    for i in range(4):
        for j in range(4):
            sub_D = D_sub[i*4:(i+1)*4, j*4:(j+1)*4]
            sub_mag = mag_sub[i*4:(i+1)*4, j*4:(j+1)*4]
            
            hist = np.zeros(8)
            for ii in range(4):
                for jj in range(4):
                    angle_deg = (np.degrees(sub_D[ii, jj]) + 360) % 360
                    bin_idx = int(angle_deg / 45) % 8
                    hist[bin_idx] += sub_mag[ii, jj]
            
            descriptor.extend(hist)
    
    descriptor = np.array(descriptor)
    norm = np.linalg.norm(descriptor)
    return descriptor / norm if norm > 0 else descriptor


def extract_descriptors(image_color, th1=50):
    keypoints = extract_keypoints(image_color, th1)
    descriptors = []
    valid_keypoints = []
    
    for kp_x, kp_y, scale, x, y, gray_img in keypoints:
        _, _, D, magnitude = calculate_gradients(gray_img)
        descriptor = calculate_descriptor(gray_img, x, y, D, magnitude)
        
        if descriptor is not None:
            descriptors.append(descriptor)
            valid_keypoints.append((kp_x, kp_y))
    
    return valid_keypoints, np.array(descriptors) if descriptors else np.array([])


def match_keypoints(desc1, desc2, threshold=17):
    if len(desc1) == 0 or len(desc2) == 0:
        return []
    
    matches = []
    for i, d1 in enumerate(desc1):
        distances = [np.sqrt(np.sum((d1 - d2)**2)) for d2 in desc2]
        min_dist = min(distances)
        min_idx = np.argmin(distances)
        
        if min_dist < threshold:
            matches.append((i, min_idx, min_dist))
    
    return matches


def draw_matches(img1, kp1, img2, kp2, matches):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    
    if h1 != h:
        scale = h / h1
        img1 = cv2.resize(img1, (int(w1 * scale), h))
        w1 = img1.shape[1]
    if h2 != h:
        scale = h / h2
        img2 = cv2.resize(img2, (int(w2 * scale), h))
        w2 = img2.shape[1]
    
    result = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    result[0:h, 0:w1] = img1
    result[0:h, w1:w1+w2] = img2
    
    for idx1, idx2, _ in matches[:50]:
        pt1 = (int(kp1[idx1][0] * h1 / h), int(kp1[idx1][1] * h1 / h))
        pt2 = (int(kp2[idx2][0] * h2 / h) + w1, int(kp2[idx2][1] * h2 / h))
        
        cv2.line(result, pt1, pt2, [0,0,255], 1)
        cv2.circle(result, pt1, 3, [0,0,255], -1)
        cv2.circle(result, pt2, 3, [0,0,255], -1)
    
    return result



I8 = cv2.imread('city.jpg')
I9 = cv2.imread('city.jpg')


th1 = 10
kp1, desc1 = extract_descriptors(I8, th1)
kp2, desc2 = extract_descriptors(I9, th1)

matches = match_keypoints(desc1, desc2) if len(kp1) > 0 and len(kp2) > 0 else []

img1_viz = I8.copy()
img2_viz = I9.copy()

for kp_x, kp_y in kp1:
    cv2.circle(img1_viz, (kp_x, kp_y), 3, (0, 255, 0), 1)
for kp_x, kp_y in kp2:
    cv2.circle(img2_viz, (kp_x, kp_y), 3, (0, 255, 0), 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(cv2.cvtColor(img1_viz, cv2.COLOR_BGR2RGB))
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(img2_viz, cv2.COLOR_BGR2RGB))
axes[1].axis('off')

if matches:
    matched_img = draw_matches(I8, kp1, I9, kp2, matches)
    axes[2].imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
else:
    combined = np.hstack([img1_viz, img2_viz])
    axes[2].imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

axes[2].axis('off')
plt.show()
