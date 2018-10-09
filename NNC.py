# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 16:59:32 2018

@author: Miriam
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
from sklearn import model_selection, metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def NNC(s,e):
    myList = []
    f=open("wdbc.data.txt")
    data_train = pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")
    X = np.array(data_train)
    
    pre_lab=X[:,1]  
    features=X[:,s:e]
    
    pre_labels=[]
    label_encoder = LabelEncoder()                                  
    pre_labels = label_encoder.fit_transform(pre_lab)
    pre_labels = np.array(pre_labels)
    
    labels=[]
    for num in range(len(pre_labels)):
        if pre_labels[num] == 0:
            labels.append([1,0])
        if pre_labels[num] == 1:
            labels.append([0,1])
    labels = np.array(labels)
    
    seq=[.9, .8, .5, .25]
    for i in seq:

        features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, train_size=i, test_size=1-i, random_state=0)
 
        features_train = np.array(features_train, dtype=np.float32)
        features_test = np.array(features_test, dtype=np.float32)
        labels_train = np.array(labels_train, dtype=np.float32)
        labels_test = np.array(labels_test, dtype=np.float32)
        
        features_train_v = Variable(torch.Tensor(features_train), requires_grad = False)
        labels_train_v = Variable(torch.Tensor(labels_train), requires_grad = False)
        features_test_v = Variable(torch.Tensor(features_test), requires_grad = False)
        labels_test_v = Variable(torch.Tensor(labels_test), requires_grad = False)
        
        
        model = nn.Sequential(nn.Linear(e-s, 15),    
                     nn.ReLU(),
                     nn.Linear(15, 10),
                     nn.ReLU(),
                     nn.Linear(10,2))
        
        loss_fn = torch.nn.SmoothL1Loss()
        optim = torch.optim.Adam(model.parameters(), lr=0.005)
        
        #training
        all_losses=[]
        for num in range(2000):
            optim.zero_grad()                                 # Intialize the hidden weights to all zeros
            pred = model(features_train_v)                    # Forward pass: Compute the output class
            loss = loss_fn(pred, labels_train_v)              # Compute the loss: difference between the output class and the                                                                   pre-given label
            all_losses.append(loss.data)
            loss.backward()                                   # Backward pass: compute the weight
            optim.step()                                      # Optimizer: update the weights of hidden nodes
            
            #print('epoch: ', num,' loss: ', loss.item())      # Print statistics
        
        #testing
        predicted_values=[]
    
        for num in range(len(features_test_v)):
            predicted_values.append(model(features_test_v[num]))
        
        score = 0
        for num in range(len(predicted_values)):
            if np.argmax(labels_test[num]) == np.argmax(predicted_values[num].data.numpy()):
                score = score + 1     
        accuracy = float(score / len(predicted_values)) * 100
        
        if accuracy <= 70:
            print ("Training:" , i*100 ,"%,"," NNC Testing Accuracy Score is: %0.2f " % (accuracy))
            print("Run it again!")
        else:
            print ("Training:", i*100 , "%,"," NNC Testing Accuracy Score is: %0.2f " % (accuracy))
    
        myList.append(accuracy)
       
    return myList 



#%%
def NNC10():
    myList = []
    f=open("wdbc.data.txt")
    data_train = pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")
    X = np.array(data_train)
    
    pre_lab=X[:,1]
    
    pre_labels=[]
    label_encoder = LabelEncoder()                                  
    pre_labels = label_encoder.fit_transform(pre_lab)
    pre_labels = np.array(pre_labels)

    features=X[:,[2,4,5,8,9,15,22,24,25,29]]
    
    labels=[]
    for num in range(len(pre_labels)):
        if pre_labels[num] == 0:
            labels.append([1,0])
        if pre_labels[num] == 1:
            labels.append([0,1])
    labels = np.array(labels)
    
    seq=[.9, .8, .5, .25]
    for i in seq:
        features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, train_size=i, test_size=1-i, random_state=0)
         
        features_train = np.array(features_train, dtype=np.float32)
        features_test = np.array(features_test, dtype=np.float32)
        labels_train = np.array(labels_train, dtype=np.float32)
        labels_test = np.array(labels_test, dtype=np.float32)    
    
        features_train_v = Variable(torch.Tensor(features_train), requires_grad = False)
        labels_train_v = Variable(torch.Tensor(labels_train), requires_grad = False)
        features_test_v = Variable(torch.Tensor(features_test), requires_grad = False)
        labels_test_v = Variable(torch.Tensor(labels_test), requires_grad = False)
        
        
        model = nn.Sequential(nn.Linear(10, 15),    
                     nn.ReLU(),
                     nn.Linear(15, 10),
                     nn.ReLU(),
                     nn.Linear(10,2))
        
        loss_fn = torch.nn.SmoothL1Loss()
        optim = torch.optim.Adam(model.parameters(), lr=0.005)
        
        #training
        all_losses=[]
        for num in range(2000):
            optim.zero_grad()                                 
            pred = model(features_train_v)                    
            loss = loss_fn(pred, labels_train_v)
            all_losses.append(loss.data)
            loss.backward()                                   
            optim.step()                                      
            #print('epoch: ', num,' loss: ', loss.item())      
        
        #testing
        predicted_values=[]
    
        for num in range(len(features_test_v)):
            predicted_values.append(model(features_test_v[num]))
        
        score = 0
        for num in range(len(predicted_values)):
            if np.argmax(labels_test[num]) == np.argmax(predicted_values[num].data.numpy()):
                score = score + 1     
        accuracy = float(score / len(predicted_values)) * 100
        
        if accuracy <= 70:
            print ("Training:" , i*100 ,"%,"," NNC Testing Accuracy Score is: %0.2f " % (accuracy))
            print("Run it again!")
        else:
            print ("Training:", i*100 , "%,"," NNC Testing Accuracy Score is: %0.2f " % (accuracy))
    
        myList.append(accuracy)
       
    return myList 

        
        