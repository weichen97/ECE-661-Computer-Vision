#!/usr/bin/env python
# coding: utf-8



import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal


def otsu(img, num_iter, img_dir, label):
    Mask = np.zeros(img.shape, dtype = np.uint8)
    channels = ['B', 'G', 'R']
    for i in range(3):
        print('img', i + 1)
        gray = img[:, :, i]
        temp = gray.flatten()
        channel = channels[i]
        for num in range(num_iter[i]):
            print('Iteration', num + 1)
            # obtain the channel's histogram
            hist, bin_edges = np.histogram(temp, bins = 256, range = (0, 256))
            
            # obtain the overall weighted pixel value
            sum_all = np.sum(hist * bin_edges[0 : -1])
            sum_back = 0
            sum_fore = 0
            num_back = 0
            num_fore = 0
            var_best = 0
            match = 0
            
            # iteratively calculate the inter-class variance over all available threshold, 
            # setting the threshold which give greatest inter-class variance
            for j in range(256):
                num_back += hist[j]
                num_fore = np.sum(hist) - num_back
                sum_back += hist[j] * j
                sum_fore = sum_all - sum_back

                if num_back != 0 and num_fore != 0:
                    avg_back = sum_back / num_back
                    avg_fore = sum_fore / num_fore
                    var = num_back * num_fore * (avg_back - avg_fore) ** 2 # calculate the inter-class variance

                    if var >= var_best:
                        var_best = var
                        match = j
            print(match)
            mask = np.zeros(gray.shape, dtype = np.uint8)
            mask[gray > match] = 1
            
            # create the input for next iteration by removing pixel larger than threshold
            temp1 = [n for n in temp if n <= match]          
            temp = temp1
            plt.imshow(mask, cmap = 'gray')
            #plt.show()
            
            cv2.imwrite(img_dir + label + '_' + channel + '_iteration' + str(num+1) +'.jpg', mask*255)
        Mask[:, :, i] = mask
    
    return Mask

def dilation(img, size, num):
    kernel = np.ones((size, size))
    mask = cv2.dilate(img, kernel, iterations = num)
    return mask

def erosion(img, size, num):
    kernel = np.ones((size, size))
    mask = cv2.erode(img, kernel, iterations = num)
    return mask

def get_contour(mask_all):
    # get the 8-neighbors to decide if a point is at contour
    contour = np.zeros(mask_all.shape, dtype = np.uint8)
    for i in range(mask_all.shape[0]):
        for j in range(mask_all.shape[1]):
            if mask_all[i, j] > 0:
                neighbor = mask_all[i-1 : i+2, j-1 : j+2]
                if np.sum(neighbor.flatten()) < 9: # not all 8-neighbors are 1 is valid contour point
                    contour[i, j] = 1
    
    return contour




def get_texture(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, dtype = np.uint8)
    mask_all = np.zeros(img.shape, dtype = np.uint8)
    layers = [3, 5, 7] # three window size 3, 5, 7
    for num in range(3):
        N = layers[num]
        edge = np.int((N - 1) / 2)
        
        # padding the original mask with edge of 0s
        temp = np.zeros((mask.shape[0] + 2*edge, mask.shape[1] + 2*edge), dtype = np.uint8)
        temp[edge:temp.shape[0]-edge, edge:temp.shape[1]-edge] = gray
        
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                x = i + edge
                y = j + edge
                window = temp[x - edge : x + edge + 1, y - edge : y + edge + 1]
                mask[i, j] = np.var(window) # assign the variance in the window to be pixel value 
        
        mask_all[:, :, num] = mask
    
    # normalize the variance value to [0, 255]
    mask_min = np.min(mask_all.flatten())
    mask_max = np.max(mask_all.flatten())
    mask_all = np.uint8(np.around((mask_all - mask_min) / (mask_max - mask_min) * 255 ))
    return mask_all



directory = "/home/xu1363/Documents/ECE 661/hw6/hw6_images/"
file1 = "cat.jpg"
file2 = "pigeon.jpeg"
file3 = "Red-Fox_.jpg"

img1 = cv2.imread(directory+file1,cv2.IMREAD_COLOR)
img2 = cv2.imread(directory+file2,cv2.IMREAD_COLOR)
img3 = cv2.imread(directory+file3,cv2.IMREAD_COLOR)



img_dir =  "/home/xu1363/Documents/ECE 661/hw6/output/img1/"
label = 'img1'
num_iter = [1, 1, 1]
mask = otsu(img1, num_iter, img_dir, label)
mask_all = mask[:, :, 0] * mask[:, :, 1] * mask[:, :, 2]

print('Combine three channels')
plt.imshow(mask_all * 255, cmap = 'gray')
plt.show()
cv2.imwrite(img_dir + label + '_combined.jpg', mask_all * 255)

mask_all = erosion(mask_all, 3, 1)
print('erosion')
plt.imshow(mask_all * 255, cmap = 'gray')
plt.show()

mask_all = dilation(mask_all, 3, 1)
print('dilation')
plt.imshow(mask_all * 255, cmap = 'gray')
plt.show()

print('Refined')
plt.imshow(mask_all * 255, cmap = 'gray')
plt.show()

contour = get_contour(mask_all)
print('Contour')
plt.imshow(contour * 255, cmap = 'gray')
plt.show()
cv2.imwrite(img_dir + label + '_contour.jpg', contour * 255)




img_dir =  "/home/xu1363/Documents/ECE 661/hw6/output/img2/"
label = 'img2'
num_iter = [1, 1, 1]
mask = otsu(img2, num_iter, img_dir, label)
mask_all = mask[:, :, 0] * mask[:, :, 1] * mask[:, :, 2]

print('Combine three channels')
plt.imshow(mask_all * 255, cmap = 'gray')
plt.show()
cv2.imwrite(img_dir + label + '_combined.jpg', mask_all * 255)

mask_all = erosion(mask_all, 5, 2)
print('erosion')
plt.imshow(mask_all * 255, cmap = 'gray')
plt.show()

mask_all = dilation(mask_all, 5, 1)
print('dilation')
plt.imshow(mask_all * 255, cmap = 'gray')
plt.show()

print('Refined')
plt.imshow(mask_all * 255, cmap = 'gray')
plt.show()

contour = get_contour(mask_all)
print('Contour')
plt.imshow(contour * 255, cmap = 'gray')
plt.show()
cv2.imwrite(img_dir + label + '_contour.jpg', contour * 255)




img_dir =  "/home/xu1363/Documents/ECE 661/hw6/output/img3/"
label = 'img3'
num_iter = [1, 1, 1]
mask = otsu(img3, num_iter, img_dir, label)
mask_all = mask[:, :, 0] * mask[:, :, 1] * mask[:, :, 2]

print('Combine three channels')
plt.imshow(mask_all * 255, cmap = 'gray')
plt.show()
cv2.imwrite(img_dir + label + '_combined.jpg', mask_all * 255)

mask_all = erosion(mask_all, 3, 1)
print('erosion')
plt.imshow(mask_all * 255, cmap = 'gray')
plt.show()

mask_all = dilation(mask_all, 3, 1)
print('dilation')
plt.imshow(mask_all * 255, cmap = 'gray')
plt.show()

print('Refined')
plt.imshow(mask_all * 255, cmap = 'gray')
plt.show()

contour = get_contour(mask_all)
print('Contour')
plt.imshow(contour * 255, cmap = 'gray')
plt.show()
cv2.imwrite(img_dir + label + '_contour.jpg', contour * 255)




# get the texture information for N = 3, 5, 7
img1_texture = get_texture(img1)
img2_texture = get_texture(img2)
img3_texture = get_texture(img3)




img_dir =  "/home/xu1363/Documents/ECE 661/hw6/output/img1/"
label = 'img1texture'
num_iter = [3, 3, 3]
mask = otsu(img1_texture, num_iter, img_dir, label)
mask_all = mask[:, :, 0] * mask[:, :, 1] * mask[:, :, 2]

print('Combine three channels')
plt.imshow(mask_all * 255, cmap = 'gray')
plt.show()
cv2.imwrite(img_dir + label + '_combined.jpg', mask_all * 255)

mask_all = erosion(mask_all, 3, 1)
print('erosion')
plt.imshow(mask_all * 255, cmap = 'gray')
plt.show()

mask_all = dilation(mask_all, 5, 2)
print('dilation')
plt.imshow(mask_all * 255, cmap = 'gray')
plt.show()

print('Refined')
plt.imshow(mask_all * 255, cmap = 'gray')
plt.show()

contour = get_contour(mask_all)
print('Contour')
plt.imshow(contour * 255, cmap = 'gray')
plt.show()
cv2.imwrite(img_dir + label + '_contour.jpg', contour * 255)




img_dir =  "/home/xu1363/Documents/ECE 661/hw6/output/img2/"
label = 'img2texture'
num_iter = [2, 2, 2]
mask = otsu(img2_texture, num_iter, img_dir, label)
mask_all = mask[:, :, 0] * mask[:, :, 1] * mask[:, :, 2]

print('Combine three channels')
plt.imshow(mask_all * 255, cmap = 'gray')
plt.show()
cv2.imwrite(img_dir + label + '_combined.jpg', mask_all * 255)

#mask_all = erosion(mask_all, 5, 1)
print('erosion')
plt.imshow(mask_all * 255, cmap = 'gray')
plt.show()

mask_all = dilation(mask_all, 5, 2)
print('dilation')
plt.imshow(mask_all * 255, cmap = 'gray')
plt.show()


print('Refined')
plt.imshow(mask_all * 255, cmap = 'gray')
plt.show()

contour = get_contour(mask_all)
print('Contour')
plt.imshow(contour * 255, cmap = 'gray')
plt.show()
cv2.imwrite(img_dir + label + '_contour.jpg', contour * 255)




img_dir =  "/home/xu1363/Documents/ECE 661/hw6/output/img3/"
label = 'img3texture'
num_iter = [3, 2, 3]
mask = otsu(img3_texture, num_iter, img_dir, label)
mask_all = mask[:, :, 0] * mask[:, :, 1] * mask[:, :, 2]

print('Combine three channels')
plt.imshow(mask_all * 255, cmap = 'gray')
plt.show()
cv2.imwrite(img_dir + label + '_combined.jpg', mask_all * 255)



mask_all = dilation(mask_all, 3, 2)
print('dilation')
plt.imshow(mask_all * 255, cmap = 'gray')
plt.show()

mask_all = erosion(mask_all, 5, 2)
print('erosion')
plt.imshow(mask_all * 255, cmap = 'gray')
plt.show()

print('Refined')
plt.imshow(mask_all * 255, cmap = 'gray')
plt.show()

contour = get_contour(mask_all)
print('Contour')
plt.imshow(contour * 255, cmap = 'gray')
plt.show()
cv2.imwrite(img_dir + label + '_contour.jpg', contour * 255)











