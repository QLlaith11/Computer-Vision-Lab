import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def get_resized(img, sizes):
	
	return [cv.resize(img,(size,size)) for size in sizes]

def get_gray(resized):
	
	return[cv.cvtColor(resized[i], cv.COLOR_BGR2GRAY) for i in range(len(resized))]

def get_octave(img):
	
	k = np.sqrt(2)
	sigma1 = 2
	sigmas = [sigma1 * (k**i) for i in range(4)]
	octaves = []
	for sig in sigmas:
		
		size = int(6 * sig)
		if size % 2 == 0:
			size += 1
		octaves.append(cv.GaussianBlur(img, (size, size), sig))
	return octaves
def get_dog(img):
	
	return [img[i+1].astype(np.float32) - img[i].astype(np.float32) for i in range(len(img)-1)]	

def get_extrema(dog):
	
	mid = dog[1]
	upper = dog[0]
	lower = dog[2]
	extrema = np.zeros_like(mid)
	
	for i in range(1, mid.shape[0] - 1):
		for j in range(1, mid.shape[1] - 1):
			
			rmid = mid[i-1:i+2,j-1:j+2]
			rup = upper[i-1:i+2,j-1:j+2]
			rlow =lower[i-1:i+2,j-1:j+2]
			pex = mid[i,j]
			vals = np.concatenate([rmid.flatten(), rup.flatten(), rlow.flatten()])
			if pex == max(vals) or pex == min(vals):
				extrema[i,j] = 1
	return extrema	
		
def mul_dog(extrema, dog):
	return extrema * dog	
	
def get_abs(img):
	return np.abs(img)

def get_thresh(img, thresh):
	
	return np.where(img > thresh, img, 0)

def get_keypoints(ext, sizes):

	base_size = 400
	all_keypoints = []
	scales = [base_size / size for size in sizes]
	for keypts, scale in zip(ext, scales):
		
		y_coords, x_coords = np.nonzero(keypts)
		keypoints = []
		for y,x in zip(y_coords, x_coords):
			scaled_x = int(x * scale)
			scaled_y = int(y * scale)
			keypoints.append((scaled_x, scaled_y))
		all_keypoints.append(keypoints)
	return all_keypoints, scales

def draw_sift(img, all_keypoints, scales):
	
	radii = [3,9,13,17]
	img_cpy = img.copy()
	for idx, keypoints in enumerate(all_keypoints):
		
		for (x,y) in keypoints:
			if 0 <= x < img.shape[1] and 0 <= y <= img.shape[0]:
				cv.circle(img_cpy, (x,y), radii[idx], [255,0,0], 2)
	return img_cpy
	
		
img = cv.imread('I0.jpg')

sizes = [400, 200, 100, 50]

resized = get_resized(img, sizes)

gray_imgs = get_gray(resized)

all_extrema = []
all_keypoints = []
all_dogs = []
for gray in gray_imgs:
	
	octaves = get_octave(gray)
	dogs = get_dog(octaves)
	all_dogs.append(dogs)
	extrema = get_extrema(dogs)
	extrema_dog = mul_dog(extrema, dogs[1])
	extrema_abs = get_abs(extrema_dog)
	extrema_thresh = get_thresh(extrema_abs, 10)
	all_extrema.append(extrema_thresh)

all_keypoints, scales = get_keypoints(all_extrema, sizes)

sift_img = draw_sift(resized[0], all_keypoints, scales)

sift_img = cv.cvtColor(sift_img, cv.COLOR_BGR2RGB)

plt.imshow(sift_img)
plt.show()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
