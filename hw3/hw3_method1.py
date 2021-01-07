#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 
import matplotlib.pyplot as plt
import numpy as np
import timeit


def get_homography(target,source):
    A = np.zeros((8,8))
    b = np.zeros((8,1))
    for i in range(4):
        x1 = source[i,0]
        y1 = source[i,1]
        x2 = target[i,0]
        y2 = target[i,1]

        A[i*2,:] = [x1,y1,1,0,0,0,-x1*x2,-y1*x2]
        A[i*2+1,:] = [0,0,0,x1,y1,1,-x1*y2,-y1*y2]

        b[i*2] = x2
        b[i*2+1] = y2

    h = np.dot(np.linalg.inv(A),b)
    h = np.append(h,1)
    H = np.reshape(h,(3,3))
    return H



def mapping(img_target,H):

    P_distort = np.array([0,0,1])
    Q_distort = np.array([0,img_target.shape[0]-1,1])
    R_distort = np.array([img_target.shape[1]-1,img_target.shape[0]-1,1])
    S_distort = np.array([img_target.shape[1]-1,0,1])


    P_world = np.matmul(H,P_distort)
    P_world = P_world / P_world[2]
    Q_world = np.matmul(H,Q_distort)
    Q_world = Q_world / Q_world[2]
    R_world = np.matmul(H,R_distort)
    R_world = R_world / R_world[2]
    S_world = np.matmul(H,S_distort)
    S_world = S_world / S_world[2]

    xmin = np.int32(np.round(np.amin([P_world[0],Q_world[0],R_world[0],S_world[0]])))
    xmax = np.int32(np.ceil(np.amax([P_world[0],Q_world[0],R_world[0],S_world[0]])))
    ymin = np.int32(np.round(np.amin([P_world[1],Q_world[1],R_world[1],S_world[1]])))
    ymax = np.int32(np.ceil(np.amax([P_world[1],Q_world[1],R_world[1],S_world[1]])))

    xlen = xmax-xmin
    ylen = ymax-ymin

    img_new = np.zeros((ylen,xlen,3), dtype=np.uint8)
    #print(xmin,ymin)
    print('The output image size is',xlen,ylen)
    Hinv = np.linalg.inv(H)

    for i in range(xlen):
        for j in range(ylen):
            input = np.array([i+xmin,j+ymin,1])
            output = np.matmul(Hinv,input)
            x = np.int(np.round(output[0]/output[2]))
            y = np.int(np.round(output[1]/output[2]))
            
            if x>0 and x<img_target.shape[1]-1 and y>0 and y<img_target.shape[0]-1:
                img_new[j,i,:] = img_target[y,x,:]
    return img_new


# In[2]:


img1 = np.array([[642, 498], [642, 532], [666, 537], [666, 503]])
img1_world = np.array([[0, 0], [0, 85], [75, 85], [75, 0]])

directory = "/home/xu1363/Documents/ECE 661/hw3/hw3_Task1_Images/Images/"
file_target = "Img1.JPG"
img_target = cv2.imread(directory+file_target,cv2.IMREAD_COLOR)

H = get_homography(img1_world,img1)
img_new = mapping(img_target,H)

plt.imshow(cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite("/home/xu1363/Documents/ECE 661/hw3/"+"img1_method1.jpeg", img_new) 


# In[3]:


img2 = np.array([[480, 722], [481, 874], [606, 923], [600, 739]])
img2_world = np.array([[0, 0], [0, 74], [84, 74], [84, 0]])

directory = "/home/xu1363/Documents/ECE 661/hw3/hw3_Task1_Images/Images/"
file_target = "Img2.jpeg"
img_target = cv2.imread(directory+file_target,cv2.IMREAD_COLOR)

H = get_homography(img2_world,img2)
img_new = mapping(img_target,H)

plt.imshow(cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB))
plt.show()

cv2.imwrite("/home/xu1363/Documents/ECE 661/hw3/"+"img2_method1.jpeg", img_new) 


# In[4]:


img3 = np.array([[2060, 700], [2092, 1483], [2695, 1333], [2666, 720]])
img3_world = np.array([[0, 0], [0, 36], [55, 36], [55, 0]])

directory = "/home/xu1363/Documents/ECE 661/hw3/hw3_Task1_Images/Images/"
file_target = "Img3.JPG"
img_target = cv2.imread(directory+file_target,cv2.IMREAD_COLOR)

H = get_homography(img3_world,img3)
img_new = mapping(img_target,H)

plt.imshow(cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite("/home/xu1363/Documents/ECE 661/hw3/"+"img3_method1.jpeg", img_new) 


# In[5]:


myimg1 = np.array([[597, 293], [589, 525], [822, 575], [820, 371]])
myimg1_world = np.array([[0, 0], [0, 25], [32, 25], [32, 0]])

directory = "/home/xu1363/Documents/ECE 661/hw3/"
file_target = "mypic1.jpg"
img_target = cv2.imread(directory+file_target,cv2.IMREAD_COLOR)

H = get_homography(myimg1_world,myimg1)
img_new = mapping(img_target,H)

plt.imshow(cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite("/home/xu1363/Documents/ECE 661/hw3/"+"myimg1_method1.jpeg", img_new) 


# In[11]:


myimg2 = np.array([[466, 64], [329, 690], [770, 755], [839, 193]])
myimg2_world = np.array([[0, 0], [0, 48], [33, 48], [33, 0]])

directory = "/home/xu1363/Documents/ECE 661/hw3/"
file_target = "mypic2.jpg"
img_target = cv2.imread(directory+file_target,cv2.IMREAD_COLOR)

H = get_homography(myimg2_world,myimg2)
img_new = mapping(img_target,H)

plt.imshow(cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite("/home/xu1363/Documents/ECE 661/hw3/"+"myimg2_method1.jpeg", img_new) 


# In[ ]:




