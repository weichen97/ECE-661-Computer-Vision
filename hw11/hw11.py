#!/usr/bin/env python
# coding: utf-8



import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt




def PCA(dir_train, dir_test, K=20):
    # get train data's features using PCA
    files = os.listdir(dir_train)
    N = len(files)
    vec_train = []
    for i in range(N):
        img = cv2.imread(dir_train + files[i])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_vec = gray.flatten()
        img_vec = img_vec / np.linalg.norm(img_vec)
        vec_train.append(img_vec)
    
    vec_train = np.array(vec_train).transpose()
    mean_train = np.mean(vec_train, axis=1)
    vec_train = vec_train - mean_train.reshape(vec_train.shape[0], -1)
    
    # get train data's labels, test data has same labels
    labels = []
    for i in range(30):
        for j in range(21):
            labels.append(i+1)
    labels = np.array(labels)
    
    # get test data's features
    files = os.listdir(dir_test)
    N = len(files)
    vec_test = []
    for i in range(N):
        img = cv2.imread(dir_test + files[i])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_vec = gray.flatten()
        img_vec = img_vec / np.linalg.norm(img_vec)
        vec_test.append(img_vec)
    
    vec_test = np.array(vec_test).transpose()
    mean_test = np.mean(vec_test, axis=1)
    vec_test = vec_test - mean_test.reshape(vec_test.shape[0], -1)
    
    # tricky way to get eigenvectors 
    d, u = np.linalg.eig(np.dot(vec_train.transpose(), vec_train))
    idx_sorted = np.argsort(d)[::-1] # biggest eigen value first
    U = u[:, idx_sorted]
    w = np.dot(vec_train, U)
    w = w / np.linalg.norm(w, axis=0)
    
    accuracys = []
    # iterate with different p value
    for p in range(K):
        result = np.zeros((len(labels), 1))
        # get subspace of p+1 dimensions
        sub = w[:, :p+1]
        features_train = np.dot(sub.transpose(), vec_train)
        features_test = np.dot(sub.transpose(), vec_test)
        KNN = KNeighborsClassifier(n_neighbors=1)
        KNN.fit(features_train.transpose(), labels)
        preds = KNN.predict(features_test.transpose())
        result[preds == labels] = 1
        accuracys.append(np.sum(result)/ result.shape[0])
        
    return accuracys




def LDA(dir_train, dir_test, K=20):
    # get train data's features using LDA
    files = os.listdir(dir_train)
    N = len(files)
    N_class = 30
    vec_train = []
    for i in range(N):
        img = cv2.imread(dir_train + files[i])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_vec = gray.flatten()
        img_vec = img_vec / np.linalg.norm(img_vec)
        vec_train.append(img_vec)
    
    vec_train = np.array(vec_train).transpose()
    mean_train = np.mean(vec_train, axis=1)
    
    # get train data's labels
    labels = []
    for i in range(30):
        for j in range(21):
            labels.append(i+1)
    labels = np.array(labels)
    
    # get test data's features
    files = os.listdir(dir_test)
    N = len(files)
    vec_test = []
    for i in range(N):
        img = cv2.imread(dir_test + files[i])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_vec = gray.flatten()
        img_vec = img_vec / np.linalg.norm(img_vec)
        vec_test.append(img_vec)
    
    vec_test = np.array(vec_test).transpose()
    mean_test = np.mean(vec_test, axis=1)
    
    mean_class = np.zeros((vec_test.shape[0], N_class))
    X = np.zeros(vec_train.shape)
    for i in range(N_class):
        mean_class[:, i] = np.mean(vec_train[:, i*20:(i+1)*20], axis=1)
        X[:, i*20:(i+1)*20] = vec_train[:, i*20:(i+1)*20] 
        \- mean_class[:, i].reshape(X.shape[0] ,-1)
        
    M = mean_class - mean_train.reshape(vec_test.shape[0], -1)
    # eigen decomposition of between class scatter SB
    d, u = np.linalg.eig(np.dot(M.transpose(), M))
    idx_sorted = np.argsort(d)[::-1] # biggest eigen value first
    D = d[idx_sorted]
    U = u[:, idx_sorted]
    v = np.dot(M, U)
    V = v / np.linalg.norm(v, axis=0)
    DB = np.zeros((N_class, N_class))
    for i in range(N_class):
        DB[i,i] = D[i]**(-0.5)
    Z = np.dot(V, DB)
    temp = np.dot(Z.transpose(), X)
    # eigen decomposition of ZSWZ
    d, u = np.linalg.eig(np.dot(temp, temp.transpose()))
    idx_sorted = np.argsort(d) # smallest eigen value first
    U = u[:, idx_sorted]
    
    accuracys = []
    # iterate with different p value
    for p in range(K):
        result = np.zeros((len(labels), 1))
        # get subspace of p+1 dimensions
        Wp = U[:, :p+1]
        sub = np.dot(Z, Wp) 
        sub = sub / np.linalg.norm(sub, axis=0)
        features_train = np.dot(sub.transpose(),   
        \vec_train - mean_train.reshape(vec_train.shape[0], -1))
        features_test = np.dot(sub.transpose(),                                
        \vec_test - mean_test.reshape(vec_test.shape[0], -1))
        KNN = KNeighborsClassifier(n_neighbors=1)
        KNN.fit(features_train.transpose(), labels)
        preds = KNN.predict(features_test.transpose())
        result[preds == labels] = 1
        accuracys.append(np.sum(result)/ result.shape[0])
        
    return accuracys




dir_train = '/home/xu1363/Documents/ECE 661/hw11/task1/train/'
dir_test = '/home/xu1363/Documents/ECE 661/hw11/task1/test/'
accuracys_PCA = PCA(dir_train, dir_test, K=20)
accuracys_LDA = LDA(dir_train, dir_test, K=20)

plt.plot(np.arange(1, 20+1), accuracys_PCA, 'b', label='PCA')
plt.plot(np.arange(1, 20+1), accuracys_LDA, 'r', label='LDA')

for i in range(20):
    plt.plot(i+1, accuracys_PCA[i], 'b*')
for i in range(20):
    plt.plot(i+1, accuracys_LDA[i], 'r+')
    
plt.legend()
plt.xlabel('Number of dimensions')
plt.ylabel('Accuracy')
plt.title('Accuracy of PCA and LDA with increasing dimension (1-20)')
plt.xlim(0,20)
plt.ylim(0,1.1)
plt.savefig('/home/xu1363/Documents/ECE 661/hw11/accuracy_task1.png')






