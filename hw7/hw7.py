#!/usr/bin/env python
# coding: utf-8



import numpy as np
import cv2
import time
import os
from BitVector import BitVector
import statistics 
from statistics import mode 




def get_inter(A, B, C, D):
    # use bilinear interpretation to calculate fractional location's intensity
    dx,dy = np.sqrt(2)/2, np.sqrt(2)/2
    return (1-dx)*(1-dy)*A + dx*(1-dy)*B + (1-dx)*dy*C + dx*dy*D

def get_bitvec(patch):
    # return the 8 neighbors intensity using the 3*3 image patch
    v1 = patch[2,1]
    v2 = get_inter(patch[1,1], patch[1,2], patch[2,1], patch[2,2])
    v3 = patch[1,2]
    v4 = get_inter(patch[1,1], patch[1,2], patch[0,1], patch[0,2])
    v5 = patch[0,1]
    v6 = get_inter(patch[1,1], patch[1,0], patch[0,0], patch[0,0])
    v7 = patch[1,0]
    v8 = get_inter(patch[1,1], patch[1,0], patch[2,1], patch[2,0])
    
    vec = np.array([v1, v2, v3, v4, v5, v6, v7, v8])
    vec = vec >= patch[1,1] #threshold according to the center pixel value
    return vec.astype(int)

def get_pattern(vec):
    # get the encoding of local binary vector based on rotation-invariant feature
    # reference: Avinash Kak, Measuring Texture and Color in Images
    bv = BitVector.BitVector( bitlist = vec ) 
    intvals_for_circular_shifts = [int(bv << 1) for _ in range(len(vec)] 
    minbv = BitVector.BitVector( intVal = min(intvals_for_circular_shifts), size = len(vec))
    bvruns = minbv.runs() 
    
    if len(bvruns) > 2: 
        return 9
    elif len(bvruns) == 1 and bvruns[0][0] == ’1’: 
        return 8
    elif len(bvruns) == 1 and bvruns[0][0] == ’0’: 
        return 0
    else: 
        return len(bvruns[1])

def LBP(img):
    # Whole procedure of obtaining the P+2 bins LBP feature histogram
    R = 1
    P = 8
    hist = [0,0,0,0,0,0,0,0,0,0]
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):   
            vec = get_bitvec(img[i-1: i+2, j-1: j+2])
            idx = get_pattern(vec)
            hist[idx] += 1 
    
    hist = np.array(hist).astype(np.float32) / np.sum(hist)
    return hist.tolist()

def train(labels):
    training_path = '/home/xu1363/Documents/ECE 661/hw7/imagesDatabaseHW7/training/'
    for label in labels:
        path = training_path + label + '/'
        hist_all = []
        for name in os.listdir(path):
            img = cv2.imread(path + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist = LBP(img)
            hist_all.append(hist)
            print(label + name + ' is finished')
        with open(training_path + label + '.txt', 'w') as filehandle:
            for i in range(len(hist_all)):
                content = ''
                hist = hist_all[i]
                for j in range(len(hist) - 1):
                    content += str(hist[j]) + ','
                content += str(hist[j+1]) + '\n'
                filehandle.write(content)

def test(labels):
    matrix_confusion = np.zeros((5,5))
    testing_path = '/home/xu1363/Documents/ECE 661/hw7/imagesDatabaseHW7/testing/'
    for i in range(len(labels)):
        test_label = labels[i]
        for num in range(5):
            name = test_label + '_' + str(num+1) + '.jpg'     
            #print(name)
            img = cv2.imread(testing_path + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist_test = LBP(img)
            
            values = []
            
            # compare current image's feature vector with all training images' feature vectors
            for label in labels:
                training_path = '/home/xu1363/Documents/ECE 661/hw7/imagesDatabaseHW7/training/'    
                with open(training_path + label +'.txt', 'r') as fh:
                    lines = fh.readlines()
                    for k in range(len(lines)):
                        line = lines[k]
                        temp = line.split(',')
                        a = []
                        for j in range(len(temp)):
                            a.append(float(temp[j]))
                        a = np.array(a)
                        b = np.array(hist_test)
                        value = np.linalg.norm(a-b)
                        values.append(value)
                        
            # sort from the minimum Euclidean distance
            values_sorted = sorted(values)    
            val_min = values_sorted[0]
            idx = values.index(val_min)
            pred = int(idx/20)
            print(i, pred)
            print('input is ' + test_label + ', pred is ' + labels[pred])
            matrix_confusion[i, pred] += 1
    
    return matrix_confusion


labels = ['beach', 'building', 'car', 'mountain', 'tree']
train(labels)
matrix_confusion = test(labels)
print(matrix_confustion)