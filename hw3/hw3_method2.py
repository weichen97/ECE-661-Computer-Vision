#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[4]:


def get_projective(corners):
    l11 = np.cross(corners[0], corners[1])
    l12 = np.cross(corners[2], corners[3])
    pt1 = np.cross(l11, l12) 
    pt1 = pt1 / pt1[2]
    
    l21 = np.cross(corners[0], corners[3])
    l22 = np.cross(corners[1], corners[2])
    pt2 = np.cross(l21, l22) 
    pt2 = pt2 / pt2[2]

    VL = np.cross(pt1, pt2) 
    VL = VL / VL[2]
    H = np.array([[1, 0, 0], [0, 1, 0], [VL[0], VL[1], 1]])

    return H

def get_affine(corners):
    l1 = np.cross(corners[0], corners[1]) / np.cross(corners[0], corners[1])[2]
    l2 = np.cross(corners[1], corners[2]) / np.cross(corners[1], corners[2])[2]
    m1 = np.cross(corners[0], corners[3]) / np.cross(corners[0], corners[3])[2]
    m2 = np.cross(corners[2], corners[3]) / np.cross(corners[2], corners[3])[2]

    A = np.array([[l1[0] * m1[0], l1[0] * m1[1] + l1[1] * m1[0]], [l2[0] * m2[0], l2[0] * m2[1] + l2[1] * m2[0]]])
    b = np.array([[-l1[1] * m1[1]], [-l2[1] * m2[1]]])
    S = np.zeros((2, 2), dtype=float)
    S[0][0] = np.dot(np.linalg.pinv(A), b)[0]
    S[0][1] = np.dot(np.linalg.pinv(A), b)[1]
    S[1][0] = np.dot(np.linalg.pinv(A), b)[1]
    S[1][1] = 1

    U, D, V = np.linalg.svd(S, full_matrices=True)
    sol = np.dot(np.dot(U, np.sqrt(np.diag(D))), U.transpose())
    H = np.array([[sol[0][0], sol[0][1], 0], [sol[1][0], sol[1][1], 0], [0, 0, 1]])

    return H


# In[5]:


# img 1


PQRS = np.array([[592,208],[585,522],[876,575],[864,297]])
PQRS_HC = np.array([[592,208,1],[585,522,1],[876,575,1],[864,297,1]])

directory = "/home/xu1363/Documents/ECE 661/hw3/hw3_Task1_Images/Images/"
file_target = "Img1.JPG"
img_target = cv2.imread(directory+file_target,cv2.IMREAD_COLOR)

H1 = get_projective(PQRS_HC)
output1 = mapping(img_target, H1)
plt.imshow(cv2.cvtColor(output1, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite("/home/xu1363/Documents/ECE 661/hw3/"+"img1_method2_projective.jpeg", output1) 


# In[6]:


H2 = get_affine(PQRS_HC)
output2 = mapping(img_target, np.dot(np.linalg.pinv(H2), H1))
plt.imshow(cv2.cvtColor(output2, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite("/home/xu1363/Documents/ECE 661/hw3/"+"img1_method2_affine.jpeg", output2) 


# In[7]:


# img 2


PQRS = np.array([[367, 553], [362, 853], [641, 975], [621, 508]])
PQRS_HC = np.array([[367, 553, 1], [362, 853, 1], [641, 975, 1], [621, 508, 1]])

directory = "/home/xu1363/Documents/ECE 661/hw3/hw3_Task1_Images/Images/"
file_target = "Img2.jpeg"
img_target = cv2.imread(directory+file_target,cv2.IMREAD_COLOR)


H1 = get_projective(PQRS_HC)
output1 = mapping(img_target, H1)
plt.imshow(cv2.cvtColor(output1, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite("/home/xu1363/Documents/ECE 661/hw3/"+"img2_method2_projective.jpeg", output1) 


# In[8]:


H2 = get_affine(PQRS_HC)
output2 = mapping(img_target, np.dot(np.linalg.pinv(H2), H1))
plt.imshow(cv2.cvtColor(output2, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite("/home/xu1363/Documents/ECE 661/hw3/"+"img2_method2_affine.jpeg", output2) 


# In[40]:


# img 3


PQRS = np.array([[2087, 739], [2120, 1429], [2673, 1302], [2651, 749]])
PQRS_HC = np.array([[2087, 739, 1], [2120, 1429, 1], [2673, 1302, 1], [2651, 749, 1]])

directory = "/home/xu1363/Documents/ECE 661/hw3/hw3_Task1_Images/Images/"
file_target = "Img3.JPG"
img_target = cv2.imread(directory+file_target,cv2.IMREAD_COLOR)

H1 = get_projective(PQRS_HC)
output1 = mapping(img_target, H1)
plt.imshow(cv2.cvtColor(output1, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite("/home/xu1363/Documents/ECE 661/hw3/"+"img3_method2_projective.jpeg", output1) 


# In[ ]:


H2 = get_affine(PQRS_HC)
output2 = mapping(img_target, np.dot(np.linalg.pinv(H2), H1))
plt.imshow(cv2.cvtColor(output2, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite("/home/xu1363/Documents/ECE 661/hw3/"+"img3_method2_affine.jpeg", output2) 


# In[10]:


# myimg 1


PQRS = np.array([[597, 293], [589, 525], [822, 575], [820, 371]])
PQRS_HC = np.array([[597, 293, 1], [589, 525, 1], [822, 575, 1], [820, 371, 1]])

directory = "/home/xu1363/Documents/ECE 661/hw3/"
file_target = "mypic1.jpg"
img_target = cv2.imread(directory+file_target,cv2.IMREAD_COLOR)

H1 = get_projective(PQRS_HC)
output1 = mapping(img_target, H1)
plt.imshow(cv2.cvtColor(output1, cv2.COLOR_BGR2RGB))
plt.show()
plt.clf()
cv2.imwrite("/home/xu1363/Documents/ECE 661/hw3/"+"myimg1_method2_projective.jpeg", output1) 


# In[11]:


H2 = get_affine(PQRS_HC)
output2 = mapping(img_target, np.dot(np.linalg.pinv(H2), H1))
plt.imshow(cv2.cvtColor(output2, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite("/home/xu1363/Documents/ECE 661/hw3/"+"myimg1_method2_affine.jpeg", output2) 


# In[12]:


# myimg 2


PQRS = np.array([[466, 64], [329, 690], [770, 755], [839, 193]])
PQRS_HC = np.array([[466, 64, 1], [329, 690, 1], [770, 755, 1], [839, 193, 1]])

directory = "/home/xu1363/Documents/ECE 661/hw3/"
file_target = "mypic2.jpg"
img_target = cv2.imread(directory+file_target,cv2.IMREAD_COLOR)

H1 = get_projective(PQRS_HC)
output1 = mapping(img_target, H1)
plt.imshow(cv2.cvtColor(output1, cv2.COLOR_BGR2RGB))
plt.show()
plt.clf()
cv2.imwrite("/home/xu1363/Documents/ECE 661/hw3/"+"myimg2_method2_projective.jpeg", output1) 


# In[13]:


H2 = get_affine(PQRS_HC)
output2 = mapping(img_target, np.dot(np.linalg.pinv(H2), H1))
plt.imshow(cv2.cvtColor(output2, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite("/home/xu1363/Documents/ECE 661/hw3/"+"myimg2_method2_affine.jpeg", output2) 


# In[ ]:




