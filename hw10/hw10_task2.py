


import numpy as np
import cv2
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt



def block_xor(block1, block2):
    center1 = block1[1, 1]
    center2 = block2[1, 1]
    line1 = block1.flatten()
    line2 = block2.flatten()

    for i in range(len(line1)):
        if center1 < line1[i]:
            line1[i] = 1
        else: line1[i] = 0
            
    for i in range(len(line2)):
        if center2 < line2[i]:
            line2[i] = 1
        else: line2[i] = 0
    
    cost = 0
    for i in range(len(line1)):
        if line1[i] != line2[i]:
            cost += 1
            
    return cost


file_truth = 'left_truedisp.pgm'
img_truth = plt.imread(path+file_truth)
img_truth = np.array(img_truth, dtype = np.float32)
img_truth /= 16
img_truth = np.array(img_truth, dtype = np.int16)
print(np.max(img_truth))

plt.imshow(img_truth, cmap = 'gray')
print(img_truth.shape)


path = '/home/xu1363/Documents/ECE 661/hw10/Task2_Images/'
file1 = 'Left.ppm'
file2 = 'Right.ppm'

img1 = plt.imread(path+file1)
img2 = plt.imread(path+file2)
gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

dmax = 14
M = 3
edge = int((M-1)/2)


h = img1.shape[0]
w = img1.shape[1]

img1new = np.zeros((h+2*edge, w+2*edge))
img2new = np.zeros((h+2*edge, w+2*edge))
img1new[edge:-edge,edge:-edge] = gray1
img2new[edge:-edge,edge:-edge] = gray2

disparity_map = np.zeros((h, w))
for j in range(h):
    for i in range(w):
        block1 = img1new[j:j+M, i:i+M]
        candidate = 0
        cost_min = 100
                         
        for k in range(dmax+1):
            i2 = i - k
            if i2 > 0:
                block2 = img2new[j:j+M, i2:i2+M]
                cost = block_xor(block1, block2)
                if cost < cost_min:
                    candidate = k
                    cost_min = cost
        disparity_map[j, i] = candidate 


img = np.array(disparity_map, dtype = np.float32)
img = img / np.max(img) * 255
img = np.array(img, dtype = np.uint16)

plt.imshow(disparity_map, cmap = 'gray')

cv2.imwrite(path + 'disparity_map_'+ str(M) + '.jpg', img)

print(np.max(img))


img_dif = abs(disparity_map - img_truth)
img_mask = np.zeros((h,w))
true = 0
for j in range(h):
    for i in range(w):
        if img_dif[j, i] <= 1:
            true += 1
            img_mask[j, i] = 255
print('accuracy is ', true/h/w)


plt.imshow(img_mask, cmap = 'gray')
cv2.imwrite(path + 'img_mask_'+ str(M) + '.jpg', img_mask)





