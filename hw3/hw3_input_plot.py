#!/usr/bin/env python
# coding: utf-8

# In[15]:


import cv2 
import matplotlib.pyplot as plt
import numpy as np
import timeit


# In[16]:


directory = "/home/xu1363/Documents/ECE 661/hw3/hw3_Task1_Images/Images/"
file_target = "Img1.JPG"
img_target = cv2.imread(directory+file_target,cv2.IMREAD_COLOR)
plt.figure(dpi=1200)
plt.imshow(cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB))

point1 = ["P1", "Q1", "R1", "S1", "P2", "Q2", "R2"]
PQRS = np.array([[592,208],[585,522],[876,575],[864,297],[168,137],[115,587],[393,622]])

for i in range(4):
    plt.scatter(PQRS[i][0], PQRS[i][1], color = 'red')
    plt.annotate(point1[i], (PQRS[i][0], PQRS[i][1]), color = 'black')
    if i < 3:
        plt.plot([PQRS[i][0], PQRS[i+1][0]], [PQRS[i][1], PQRS[i+1][1]], color = 'red')
    if i == 3:
        plt.plot([PQRS[3][0], PQRS[0][0]], [PQRS[3][1], PQRS[0][1]], color = 'red')
for i in range(4,7):
    plt.scatter(PQRS[i][0], PQRS[i][1], color = 'blue')
    plt.annotate(point1[i], (PQRS[i][0], PQRS[i][1]), color = 'black')
    if i < 6:
        plt.plot([PQRS[i][0], PQRS[i+1][0]], [PQRS[i][1], PQRS[i+1][1]], color = 'blue')

PQRS = np.array([[642, 498], [642, 532], [666, 537], [666, 503]])        
point1 = ["P", "Q", "R", "S"]
for i in range(4):
    plt.scatter(PQRS[i][0], PQRS[i][1], color = 'yellow')
    plt.annotate(point1[i], (PQRS[i][0], PQRS[i][1]), color = 'black')
    if i < 3:
        plt.plot([PQRS[i][0], PQRS[i+1][0]], [PQRS[i][1], PQRS[i+1][1]], color = 'yellow')
    if i == 3:
        plt.plot([PQRS[3][0], PQRS[0][0]], [PQRS[3][1], PQRS[0][1]], color = 'yellow')   
plt.axis('off')
plt.savefig("/home/xu1363/Documents/ECE 661/hw3/"+"img1_input.jpeg")


# In[17]:


directory = "/home/xu1363/Documents/ECE 661/hw3/hw3_Task1_Images/Images/"
file_target = "Img2.jpeg"
img_target = cv2.imread(directory+file_target,cv2.IMREAD_COLOR)
plt.figure(dpi=1200)

plt.imshow(cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB))

point1 = ["P1", "Q1", "R1", "S1", "P2", "Q2", "R2"]
PQRS = np.array([[367, 553], [362, 853], [641, 975], [621, 508], [412, 100], [409, 374], [538, 311]])

for i in range(4):
    plt.scatter(PQRS[i][0], PQRS[i][1], color = 'red')
    plt.annotate(point1[i], (PQRS[i][0], PQRS[i][1]), color = 'black')
    if i < 3:
        plt.plot([PQRS[i][0], PQRS[i+1][0]], [PQRS[i][1], PQRS[i+1][1]], color = 'red')
    if i == 3:
        plt.plot([PQRS[3][0], PQRS[0][0]], [PQRS[3][1], PQRS[0][1]], color = 'red')
for i in range(4,7):
    plt.scatter(PQRS[i][0], PQRS[i][1], color = 'blue')
    plt.annotate(point1[i], (PQRS[i][0], PQRS[i][1]), color = 'black')
    if i < 6:
        plt.plot([PQRS[i][0], PQRS[i+1][0]], [PQRS[i][1], PQRS[i+1][1]], color = 'blue')

PQRS = np.array([[480, 722], [481, 874], [606, 923], [600, 739]])        
point1 = ["P", "Q", "R", "S"]
for i in range(4):
    plt.scatter(PQRS[i][0], PQRS[i][1], color = 'yellow')
    plt.annotate(point1[i], (PQRS[i][0], PQRS[i][1]), color = 'black')
    if i < 3:
        plt.plot([PQRS[i][0], PQRS[i+1][0]], [PQRS[i][1], PQRS[i+1][1]], color = 'yellow')
    if i == 3:
        plt.plot([PQRS[3][0], PQRS[0][0]], [PQRS[3][1], PQRS[0][1]], color = 'yellow')    
plt.axis('off')
plt.savefig("/home/xu1363/Documents/ECE 661/hw3/"+"img2_input.jpeg")


# In[21]:


directory = "/home/xu1363/Documents/ECE 661/hw3/hw3_Task1_Images/Images/"
file_target = "Img3.JPG"
img_target = cv2.imread(directory+file_target,cv2.IMREAD_COLOR)
plt.figure(dpi=1200)

plt.imshow(cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB))

point1 = ["P1", "Q1", "R1", "S1", "P2", "Q2", "R2"]
PQRS = np.array([[2087, 739], [2120, 1429], [2673, 1302], [2651, 749], [746, 820], [790, 2002], [1749, 1577]])

for i in range(4):
    plt.scatter(PQRS[i][0], PQRS[i][1], color = 'red')
    plt.annotate(point1[i], (PQRS[i][0], PQRS[i][1]), color = 'black')
    if i < 3:
        plt.plot([PQRS[i][0], PQRS[i+1][0]], [PQRS[i][1], PQRS[i+1][1]], color = 'red')
    if i == 3:
        plt.plot([PQRS[3][0], PQRS[0][0]], [PQRS[3][1], PQRS[0][1]], color = 'red')
for i in range(4,7):
    plt.scatter(PQRS[i][0], PQRS[i][1], color = 'blue')
    plt.annotate(point1[i], (PQRS[i][0], PQRS[i][1]), color = 'black')
    if i < 6:
        plt.plot([PQRS[i][0], PQRS[i+1][0]], [PQRS[i][1], PQRS[i+1][1]], color = 'blue')
plt.axis('off')
plt.savefig("/home/xu1363/Documents/ECE 661/hw3/"+"img3_input.jpeg")


# In[19]:


directory = "/home/xu1363/Documents/ECE 661/hw3/"
file_target = "mypic1.jpg"
img_target = cv2.imread(directory+file_target,cv2.IMREAD_COLOR)
plt.figure(dpi=1200)

plt.imshow(cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB))

point1 = ["P1", "Q1", "R1", "S1", "P2", "Q2", "R2"]
PQRS = np.array([[597, 293], [589, 525], [822, 575], [820, 371], [1020, 442], [1032, 619], [1165, 648]])

for i in range(4):
    plt.scatter(PQRS[i][0], PQRS[i][1], color = 'red')
    plt.annotate(point1[i], (PQRS[i][0], PQRS[i][1]), color = 'black')
    if i < 3:
        plt.plot([PQRS[i][0], PQRS[i+1][0]], [PQRS[i][1], PQRS[i+1][1]], color = 'red')
    if i == 3:
        plt.plot([PQRS[3][0], PQRS[0][0]], [PQRS[3][1], PQRS[0][1]], color = 'red')
for i in range(4,7):
    plt.scatter(PQRS[i][0], PQRS[i][1], color = 'blue')
    plt.annotate(point1[i], (PQRS[i][0], PQRS[i][1]), color = 'black')
    if i < 6:
        plt.plot([PQRS[i][0], PQRS[i+1][0]], [PQRS[i][1], PQRS[i+1][1]], color = 'blue')
plt.axis('off')
plt.savefig("/home/xu1363/Documents/ECE 661/hw3/"+"myimg1_input.jpeg")


# In[20]:


directory = "/home/xu1363/Documents/ECE 661/hw3/"
file_target = "mypic2.jpg"
img_target = cv2.imread(directory+file_target,cv2.IMREAD_COLOR)
plt.figure(dpi=1200)

plt.imshow(cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB))

point1 = ["P1", "Q1", "R1", "S1", "P2", "Q2", "R2"]
PQRS = np.array([[466, 64], [329, 690], [770, 755], [839, 193], [1053, 267], [1008, 799], [1332, 860]])

for i in range(4):
    plt.scatter(PQRS[i][0], PQRS[i][1], color = 'red')
    plt.annotate(point1[i], (PQRS[i][0], PQRS[i][1]), color = 'black')
    if i < 3:
        plt.plot([PQRS[i][0], PQRS[i+1][0]], [PQRS[i][1], PQRS[i+1][1]], color = 'red')
    if i == 3:
        plt.plot([PQRS[3][0], PQRS[0][0]], [PQRS[3][1], PQRS[0][1]], color = 'red')
for i in range(4,7):
    plt.scatter(PQRS[i][0], PQRS[i][1], color = 'blue')
    plt.annotate(point1[i], (PQRS[i][0], PQRS[i][1]), color = 'black')
    if i < 6:
        plt.plot([PQRS[i][0], PQRS[i+1][0]], [PQRS[i][1], PQRS[i+1][1]], color = 'blue')
plt.axis('off')
plt.savefig("/home/xu1363/Documents/ECE 661/hw3/"+"myimg2_input.jpeg")


# In[ ]:




