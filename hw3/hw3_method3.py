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


# In[9]:


def compute_coe(array1, array2):
    output = np.array([array1[0] * array2[0], (array1[0]*array2[1]+array1[1]*array2[0])/2,
                       array1[1]*array2[2], (array1[0]*array2[2]+array1[2]*array2[0])/2,
                       (array1[1]*array2[2]+array1[2]*array2[1])/2])

    return output

'''
def H_matrix(pts):
    # PQ \ PS
    l1 = np.cross(pts[0], pts[1]) / np.max(np.cross(pts[0], pts[1]))
    m1 = np.cross(pts[1], pts[2]) / np.max(np.cross(pts[1], pts[2]))
    # PS \ SR
    l2 = np.cross(pts[1], pts[2]) / np.max(np.cross(pts[1], pts[2]))
    m2 = np.cross(pts[2], pts[3]) / np.max(np.cross(pts[2], pts[3]))
    # SR \ QR
    l3 = np.cross(pts[2], pts[3]) / np.max(np.cross(pts[2], pts[3]))
    m3 = np.cross(pts[3], pts[0]) / np.max(np.cross(pts[3], pts[0]))
    # PQ \ QR
    l4 = np.cross(pts[3], pts[0]) / np.max(np.cross(pts[3], pts[0]))
    m4 = np.cross(pts[0], pts[1]) / np.max(np.cross(pts[0], pts[1]))
    # AB \ BC
    l5 = np.cross(pts[4], pts[5]) / np.max(np.cross(pts[4], pts[5]))
    m5 = np.cross(pts[5], pts[6]) / np.max(np.cross(pts[5], pts[6]))
    # Compute S_mat
    A = np.zeros((5,5))
    A[0,:] = compute_coe(l1, m1)
    A[1,:] = compute_coe(l2, m2)
    A[2,:] = compute_coe(l3, m3)
    A[3,:] = compute_coe(l4, m4)
    A[4,:] = compute_coe(l5, m5)
    b = np.array([[-l1[2] * m1[2]], [-l2[2] * m2[2]], [-l3[2] * m3[2]], [-l4[2] * m4[2]], [-l5[2] * m5[2]]])
    sol = np.dot(np.linalg.pinv(A), b) / np.max(np.abs(np.dot(np.linalg.pinv(A), b)))
    S = np.zeros((2, 2), dtype=float)
    S[0][0] = sol[0]
    S[0][1] = sol[1] / 2
    S[1][0] = sol[1] / 2
    S[1][1] = sol[2]
    # Compute SVD of S
    S_mat = np.array(S, dtype=float)
    U, D, V = np.linalg.svd(S_mat, full_matrices=True)
    temp = np.dot(np.dot(U, np.sqrt(np.diag(D))), U.transpose())
    v_vec = np.dot(np.linalg.pinv(temp), np.array([sol[3] / 2, sol[4] / 2]))
    H = np.array([[temp[0][0], temp[0][1], 0], [temp[1][0], temp[1][1], 0], [v_vec[0], v_vec[1], 1]], dtype=float)

    return H'''
def H_matrix(pts):
    # PQ \ PS
    l1 = np.cross(pts[0], pts[1]) / np.max(np.cross(pts[0], pts[1]))
    m1 = np.cross(pts[0], pts[2]) / np.max(np.cross(pts[0], pts[2]))
    # PS \ SR
    l2 = np.cross(pts[0], pts[2]) / np.max(np.cross(pts[0], pts[2]))
    m2 = np.cross(pts[2], pts[3]) / np.max(np.cross(pts[2], pts[3]))
    # SR \ QR
    l3 = np.cross(pts[2], pts[3]) / np.max(np.cross(pts[2], pts[3]))
    m3 = np.cross(pts[1], pts[3]) / np.max(np.cross(pts[1], pts[3]))
    # PQ \ QR
    l4 = np.cross(pts[0], pts[1]) / np.max(np.cross(pts[0], pts[1]))
    m4 = np.cross(pts[1], pts[3]) / np.max(np.cross(pts[1], pts[3]))
    # AB \ BC
    l5 = np.cross(pts[4], pts[5]) / np.max(np.cross(pts[4], pts[5]))
    m5 = np.cross(pts[5], pts[6]) / np.max(np.cross(pts[5], pts[6]))
    # Compute S_mat
    A = []
    A.append(compute_coe(l1, m1))
    A.append(compute_coe(l2, m2))
    A.append(compute_coe(l3, m3))
    A.append(compute_coe(l4, m4))
    A.append(compute_coe(l5, m5))
    A = np.asarray(A)
    b = np.array([[-l1[2] * m1[2]], [-l2[2] * m2[2]], [-l3[2] * m3[2]], [-l4[2] * m4[2]], [-l5[2] * m5[2]]])
    sol = np.dot(np.linalg.pinv(A), b) / np.max(np.abs(np.dot(np.linalg.pinv(A), b)))
    S = np.zeros((2, 2), dtype=float)
    S[0][0] = sol[0]
    S[0][1] = sol[1] / 2
    S[1][0] = sol[1] / 2
    S[1][1] = sol[2]
    # Compute SVD of S
    S_mat = np.array(S, dtype=float)
    U, D, V = np.linalg.svd(S_mat, full_matrices=True)
    temp = np.dot(np.dot(U, np.sqrt(np.diag(D))), U.transpose())
    v_vec = np.dot(np.linalg.pinv(temp), np.array([sol[3] / 2, sol[4] / 2]))
    H = np.array([[temp[0][0], temp[0][1], 0], [temp[1][0], temp[1][1], 0], [v_vec[0], v_vec[1], 1]], dtype=float)

    return H


# In[11]:


directory = "/home/xu1363/Documents/ECE 661/hw3/hw3_Task1_Images/Images/"
file_target = "Img1.JPG"
img_target = cv2.imread(directory+file_target,cv2.IMREAD_COLOR)

PQSR1 = np.array([[462, 165], [431, 730], [1415, 482], [1462, 810], [165, 69], [98, 876], [425, 883]])
point1 = ["P", "Q", "S", "R", "A", "B", "C"]
plt.imshow(cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB))
for i in range(7):
    plt.scatter(PQSR1[i][0], PQSR1[i][1])
    plt.annotate(point1[i] + '({},{})'.format(PQSR1[i][0], PQSR1[i][1]), (PQSR1[i][0], PQSR1[i][1]), c='r')


# In[12]:


pts = np.array([[462, 165, 1], [431, 730, 1], [1415, 482, 1], [1462, 810, 1], [165, 69, 1], [98, 876, 1], [425, 883, 1]])
H1 = H_matrix(pts)
output1 = mapping(img_target, np.linalg.inv(H1))
plt.imshow(cv2.cvtColor(output1, cv2.COLOR_BGR2RGB))
cv2.imwrite("/home/xu1363/Documents/ECE 661/hw3/"+"img1_method3.jpeg", output1) 


# In[3]:


# img 1


PQRS = np.array([[592,208],[585,522],[876,575],[864,297],[168,137],[115,587],[393,622]])
PQRS_HC = np.array([[592,208,1],[585,522,1],[876,575,1],[864,297,1],[168,137,1],[115,587,1],[393,622,1]])

directory = "/home/xu1363/Documents/ECE 661/hw3/hw3_Task1_Images/Images/"
file_target = "Img1.JPG"
img_target = cv2.imread(directory+file_target,cv2.IMREAD_COLOR)

H1 = H_matrix(PQRS_HC)
img_new = mapping(img_target, np.linalg.inv(H1))
plt.imshow(cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB))
cv2.imwrite("/home/xu1363/Documents/ECE 661/hw3/"+"img1_method3.jpeg", img_new) 


# In[4]:


# img 2


PQRS = np.array([[367, 553], [362, 853], [641, 975], [621, 508], [412, 100], [409, 374], [538, 311]])
PQRS_HC = np.array([[367, 553, 1], [362, 853, 1], [641, 975, 1], [621, 508, 1], [412, 100, 1], [409, 374, 1], [538, 311, 1]])

directory = "/home/xu1363/Documents/ECE 661/hw3/hw3_Task1_Images/Images/"
file_target = "Img2.jpeg"
img_target = cv2.imread(directory+file_target,cv2.IMREAD_COLOR)
 
H1 = H_matrix(PQRS_HC)
img_new = mapping(img_target, np.linalg.inv(H1))
plt.imshow(cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB))
cv2.imwrite("/home/xu1363/Documents/ECE 661/hw3/"+"img2_method3.jpeg", img_new) 


# In[5]:


# img 3

PQRS = np.array([[2087, 739], [2120, 1429], [2673, 1302], [2651, 749], [746, 820], [790, 2002], [1749, 1577]])
PQRS_HC = np.array([[2087, 739, 1], [2120, 1429, 1], [2673, 1302, 1], [2651, 749, 1], [746, 820, 1], [790, 2002, 1], [1749, 1577, 1]])

directory = "/home/xu1363/Documents/ECE 661/hw3/hw3_Task1_Images/Images/"
file_target = "Img3.JPG"
img_target = cv2.imread(directory+file_target,cv2.IMREAD_COLOR)
  
H1 = H_matrix(PQRS_HC)
img_new = mapping(img_target, np.linalg.inv(H1))
plt.imshow(cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB))
cv2.imwrite("/home/xu1363/Documents/ECE 661/hw3/"+"img3_method3.jpeg", img_new) 


# In[6]:


# my own picture 1

PQRS = np.array([[597, 293], [589, 525], [822, 575], [820, 371], [1020, 442], [1032, 619], [1165, 648]])
PQRS_HC = np.array([[597, 293, 1], [589, 525, 1], [822, 575, 1], [820, 371, 1], [1020, 442, 1], [1032, 619, 1], [1165, 648, 1]])

directory = "/home/xu1363/Documents/ECE 661/hw3/"
file_target = "mypic1.jpg"
img_target = cv2.imread(directory+file_target,cv2.IMREAD_COLOR)

H1 = H_matrix(PQRS_HC)
img_new = mapping(img_target, np.linalg.inv(H1))
plt.imshow(cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB))
cv2.imwrite("/home/xu1363/Documents/ECE 661/hw3/"+"myimg1_method3.jpeg", img_new) 


# In[7]:


# my own picture 2

PQRS = np.array([[466, 64], [329, 690], [770, 755], [839, 193], [1053, 267], [1008, 799], [1332, 860]])
PQRS_HC = np.array([[466, 64, 1], [329, 690, 1], [770, 755, 1], [839, 193, 1], [1053, 267, 1], [1008, 799, 1], [1332, 860, 1]])

directory = "/home/xu1363/Documents/ECE 661/hw3/"
file_target = "mypic2.jpg"
img_target = cv2.imread(directory+file_target,cv2.IMREAD_COLOR)

H1 = H_matrix(PQRS_HC)
img_new = mapping(img_target, np.linalg.inv(H1))
plt.imshow(cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB))
cv2.imwrite("/home/xu1363/Documents/ECE 661/hw3/"+"myimg2_method3.jpeg", img_new) 


# In[ ]:




