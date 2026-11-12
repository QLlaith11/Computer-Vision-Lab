import numpy as np
import cv2 as cv



img = cv.imread("P0.jfif",cv.IMREAD_GRAYSCALE)
cv.imshow('',img)
cv.waitKey(0)
print(img.shape)
blur = cv.GaussianBlur(img, (3,3), 1)
cv.imshow('',blur)
cv.waitKey(0)
edges = cv.Canny(blur, 50, 150)
print(edges.shape)
cv.imshow('', edges)
cv.waitKey(0)
h, w = edges.shape
tb = 180
m_rho = int(np.sqrt(h**2 + w**2))
print(m_rho)

ths = np.deg2rad(np.arange(-90,90, 180/tb))

acc = np.zeros((2*m_rho, tb),dtype = np.uint64)

y_idx, x_idx = np.nonzero(edges)
ep_tot = len(x_idx)

for i in range(ep_tot):
	x = x_idx[i]
	y = y_idx[i]
	
	for th_idx, th in enumerate(ths):
		rho = x*np.cos(th) + y*np.sin(th)
		rho_idx = int(rho) + m_rho
		if 0 <=rho_idx<acc.shape[0]:
			acc[rho_idx, th_idx] += 1
			

thresh = 0.3 * np.max(acc)
rho_idxs, theta_idxs = np.where(acc >= thersh)
votes = acc[rho_idxs, theta_idxs]
f_rho = []
f_th = []
f_votes = []
used = set()
for i in range(len(rho_idxs)):
	rho = rho_idxs[i]
	theta = theta_idxs[i]
	close = False
	for used_rho,used_th in used:
		if abs(rho - used_rho)<5 and abs(theta-used_theta)<5:
			close = True
			break
	if not close:
		f_rho.append(rho)
		f_th.append(theta)
		f_votes.append(votes[i])
		used.add((rho,theta))
lines = []

for r,t in zip(f_rho,f_th):
	rho = r-m_rho
	theta = ths[t]
	a = np.cos(theta)
	b = np.sin(theta)
	points = []
	
	if abs(b) > 1e-6:
		x_top = rho/a if abs(a)>1e-6 else 0
		if 0<= x_top < w:
		points.append(int(x_top), 0)
		
		
		
		
		
		
		
		
		
		
		
		














