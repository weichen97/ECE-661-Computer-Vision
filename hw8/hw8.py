#!/usr/bin/env python
# coding: utf-8


import numpy as np
import cv2
import time
import os
import matplotlib.pyplot as plt
from scipy import signal
from sklearn import svm, metrics
from sklearn.metrics import confusion_matrix

def get_conv_mask(C):
    # create C of different convolutional masks
    masks = np.zeros((C,3,3))
    for i in range(C):
        conv_mask = np.random.uniform(-1, 1, size = (3,3))
        conv_mask = conv_mask - np.sum(conv_mask) / 9
        masks[i, :, :] = conv_mask
    return masks

def get_Gram_feature(img, masks):
    # masks contatin C different convolutional masks
    vec = []
    for i in range(masks.shape[0]):
        mask = masks[i,:,:]
        # obtain result of convolving image with mask
        img_conv = signal.convolve2d(img, mask, mode='same')
        img_output = cv2.resize(img_conv, (16, 16), interpolation = cv2.INTER_AREA)
        vec.append(img_output.flatten())
    
    vec = np.array(vec)
    gram = np.dot(vec, np.transpose(vec)) # obtain gram matrix
    
    # retain the upper triangular part of gram matrix
    vec_feature = []
    for i in range(C):
        for j in range(C):
            if j >= i:
                vec_feature.append(gram[i][j])
    
    return np.array(vec_feature)

def check_label(name):
    # obtain image label from file name
    if 'cloudy' in name:
        return 0
    if 'rain' in name:
        return 1
    if 'shine' in name:
        return 2
    if 'sunrise' in name:
        return 3

def seperate_dataset(inputs, labels):
    # seperate data into training data and validation data
    idx_arr = np.arange(len(inputs))
    np.random.shuffle(idx_arr)
    
    ratio = 0.7 # ratio of training data in all data
    idx_train, idx_test = idx_arr[: int(ratio * len(inputs))], idx_arr[int(ratio * len(inputs)) : ] 
    
    inputs_train, labels_train = inputs[idx_train, :], labels[idx_train]
    inputs_test, labels_test = inputs[idx_test, :], labels[idx_test]
    

    return inputs_train, labels_train, inputs_test, labels_test

def training(N, C):
    training_path = '/home/xu1363/Documents/ECE 661/hw8/imagesDatabaseHW8/training/'
    # resizing image to common size
    h = 128
    w = 128
    accuracy_best = 0
    
    for n in range(N):
        print('Trial #' + str(n+1))
        labels = []
        inputs = []
        masks = get_conv_mask(C)
        for name in os.listdir(training_path):
            label = check_label(name)
            img = cv2.imread(training_path + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (h, w), interpolation = cv2.INTER_AREA)
            vec_feature = get_Gram_feature(img, masks)

            labels.append(int(label))
            inputs.append(vec_feature)

        inputs = np.array(inputs)
        labels = np.array(labels)
        
        # build SVM model
        svm=cv2.ml.SVM_create()
        svm.setC(1)
        svm.setGamma(0.1)
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_LINEAR)
        #svm.setKernel(cv2.ml.SVM_RBF)
        #svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))

        # train
        inputs_train, labels_train, inputs_valid, labels_valid = seperate_dataset(inputs, labels)
        svm.train(np.float32(inputs_train), cv2.ml.ROW_SAMPLE, labels_train)
        
        _,preds =svm.predict(np.float32(np.array(inputs_train)))
        preds = preds.ravel()
        accuracy_train = metrics.accuracy_score(labels_train, preds)
        print('Training accuracy is ', accuracy_train)
        
        # validate
        _,preds =svm.predict(np.float32(np.array(inputs_valid)))
        preds = preds.ravel()
        accuracy_valid = metrics.accuracy_score(labels_valid, preds)
        print('Validation accuracy is ', accuracy_valid)
        
        # save model only when validation accuracy improves
        if accuracy_valid > accuracy_best:
            accuracy_best = accuracy_valid
            masks_best = masks
            svm.save('/home/xu1363/Documents/ECE 661/hw8/trained_svm.xml')
            print('svm has been saved')
        
        print('')
        
    # return with convolutional masks generating best validation accuracy
    return masks

def testing(masks):
    testing_path = '/home/xu1363/Documents/ECE 661/hw8/imagesDatabaseHW8/testing/'
    h = 128
    w = 128
    #C = 8
    labels = []
    inputs = []
    svm=cv2.ml.SVM_load('trained_svm.xml')
    for name in os.listdir(testing_path):
        label = check_label(name)
        img = cv2.imread(testing_path + name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (h, w), interpolation = cv2.INTER_AREA)
        vec_feature = get_Gram_feature(img, masks)
        labels.append(int(label))
        inputs.append(vec_feature)
        
    _, preds=svm.predict(np.float32(np.array(inputs)))
    preds = preds.ravel()
    print(labels)
    print(preds)
    
    accuracy = metrics.accuracy_score(labels,preds)
    print('Test accuracy is', accuracy)
    print(metrics.confusion_matrix(labels, preds))


C = 20
N = 100
masks = training(N, C)
testing(masks)

