# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 16:10:35 2018

@author: Miriam
"""

import numpy as np
import pandas as pd
from sklearn import linear_model, datasets, metrics, svm, tree, neural_network, model_selection
from sklearn import neighbors, gaussian_process, ensemble, naive_bayes, discriminant_analysis

def LogReg(s,e):
    myList = []
    f=open("wdbc.data.txt")                                                     
    data= pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")                                                                                       
    X=np.array(data)                                                            
    Y=X[:,1]                                                                                                                                            
    Y= np.where(Y=='M', 1, 0)                                                   
    X=X[:,s:e] 
    
    seq = [.9, .8, .5, .25]

    for i in seq:
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=i, test_size= 1-i, random_state=0)
        
        cl = linear_model.LogisticRegression(C=2.5)
        cl.fit(X_train,Y_train)
        Z=cl.predict(X_test)
        scores = metrics.accuracy_score(Y_test, Z)*100
        
        print("Logistic Regression. Training:" , i*100 ,"%")
        print(metrics.classification_report(Y_test,Z))
        print(metrics.confusion_matrix(Y_test,Z))
        print("Accuracy: %0.2f" % (scores))
            
        myList.append(scores)
        
    return myList
        
           
def SVM(s,e):
    myList = []
    f=open("wdbc.data.txt")
    data= pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")
    X=np.array(data)
    Y=X[:,1]
    Y= np.where(Y=='M', 1, 0)
    X=X[:,s:e]
        
    seq = [.9, .8, .5, .25]
    for i in seq:
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=i, test_size=1-i, random_state=0)
    
        cl=svm.SVC(kernel='linear')                                                           
        cl.fit(X_train,Y_train)
        Z=cl.predict(X_test)
        scores = metrics.accuracy_score(Y_test, Z)*100
        
        print("Support Vector. Training:" , i*100 ,"%")
        print(metrics.classification_report(Y_test,Z))
        print(metrics.confusion_matrix(Y_test,Z))
        print("Accuracy: %0.2f" % (scores))
    
        myList.append(scores)
        
    return myList
        
    
def DTC(s,e):
    myList = []
    f=open("wdbc.data.txt")
    data= pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")
    X=np.array(data)
    Y=X[:,1]
    Y= np.where(Y=='M', 1, 0)
    X=X[:,s:e]

    seq = [.9, .8, .5, .25]
    for i in seq:
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=i, test_size=1-i, random_state=0)
    
        cl = tree.DecisionTreeClassifier()                                                                
        cl.fit(X_train,Y_train)
        Z=cl.predict(X_test)
        scores = metrics.accuracy_score(Y_test, Z)*100
        
        print("Decision Tree. Training:" , i*100 ,"%")
        print(metrics.classification_report(Y_test,Z))
        print(metrics.confusion_matrix(Y_test,Z))
        print("Accuracy: %0.2f" % (scores))
        
        myList.append(scores)
        
    return myList

def KNC(s,e):
    myList = []
    f=open("wdbc.data.txt")
    data = pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")
    X=np.array(data)
    Y=X[:,1]
    Y=np.where(Y=='M',1,0)
    X=X[:,s:e]
    
    seq = [.9, .8, .5, .25]
    for i in seq:
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=i, test_size=1-i, random_state=0)
    
        cl = neighbors.KNeighborsClassifier(n_neighbors=3)                                                             
        cl.fit(X_train,Y_train)
        Z=cl.predict(X_test)
        scores = metrics.accuracy_score(Y_test, Z)*100
        
        print("K Neighbors. Training:" , i*100 ,"%")
        print(metrics.classification_report(Y_test,Z))
        print(metrics.confusion_matrix(Y_test,Z))
        print("Accuracy: %0.2f" % (scores))
        
        myList.append(scores)
        
    return myList


def RFC(s,e):
    myList = []
    f=open("wdbc.data.txt")
    data=pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")
    X=np.array(data)
    Y=X[:,1]
    Y=np.where(Y=='M',1,0)
    X=X[:,s:e]
    
    seq = [.9, .8, .5, .25]
    for i in seq:
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=i, test_size=1-i, random_state=0)
    
        cl=ensemble.RandomForestClassifier(max_depth=15, n_estimators=10, max_features=1)
        cl.fit(X_train,Y_train)
        Z=cl.predict(X_test)
        scores = metrics.accuracy_score(Y_test, Z)*100
        
        print("Random Forest. Training:" , i*100 ,"%")
        print(metrics.classification_report(Y_test,Z))
        print(metrics.confusion_matrix(Y_test,Z))
        print("Accuracy: %0.2f" % (scores))
        
        myList.append(scores)
        
    return myList
    
    
def MLP(s,e):
    myList = []
    f=open("wdbc.data.txt")
    data = pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")
    X=np.array(data)
    Y=X[:,1]
    Y=np.where(Y=='M',1,0)
    X=X[:, s:e]
    
    seq = [.9, .8, .5, .25]
    for i in seq:
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=i, test_size=1-i, random_state=0)
    
        cl = neural_network.MLPClassifier(activation='logistic', solver='lbfgs', max_iter=1000 )                                                                   
        cl.fit(X_train,Y_train)
        Z=cl.predict(X_test)
        scores = metrics.accuracy_score(Y_test, Z)*100
        
        print("Multi Layer Perceptron. Training:" , i*100 ,"%")
        print(metrics.classification_report(Y_test,Z))
        print(metrics.confusion_matrix(Y_test,Z))
        print("Accuracy: %0.2f" % (scores)) 
        
        myList.append(scores)
        
    return myList
    
def ABC(s,e):
    myList = []
    f=open("wdbc.data.txt")
    data=pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")
    X=np.array(data)
    Y=X[:,1]
    Y=np.where(Y=='M',1,0)
    X=X[:,s:e]
    
    seq = [.9, .8, .5, .25]
    for i in seq:
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=i, test_size=1-i, random_state=0)
    
        cl=ensemble.AdaBoostClassifier()                                                                    
        cl.fit(X_train,Y_train)
        Z=cl.predict(X_test)
        scores = metrics.accuracy_score(Y_test, Z)*100
        
        print("Ada Boost. Training:" , i*100 ,"%")
        print(metrics.classification_report(Y_test,Z))
        print(metrics.confusion_matrix(Y_test,Z))
        print("Accuracy: %0.2f" % (scores))
        
        myList.append(scores)
        
    return myList
        
    
def GNB(s,e):
    myList = []
    f=open("wdbc.data.txt")
    data=pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")
    X=np.array(data)
    Y=X[:,1]
    Y=np.where(Y=='M',1,0)
    X=X[:,s:e]
    
    seq = [.9, .8, .5, .25]
    for i in seq:
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=i, test_size=1-i, random_state=0)
    
        cl=naive_bayes.GaussianNB()
        cl.fit(X_train,Y_train)
        Z=cl.predict(X_test)
        scores = metrics.accuracy_score(Y_test, Z)*100
        
        print("Gaussian Naive Bayes. Training:" , i*100 ,"%")
        print(metrics.classification_report(Y_test,Z))
        print(metrics.confusion_matrix(Y_test,Z))
        print("Accuracy: %0.2f" % (scores))
        
        myList.append(scores)
    return myList
    
    
def QDA(s,e):
    myList = []
    f=open("wdbc.data.txt")
    data=pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")
    X=np.array(data)
    Y=X[:,1]
    Y=np.where(Y=='M',1,0)
    X=X[:,s:e]
    
    seq = [.9, .8, .5, .25]
    for i in seq:
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=i, test_size=1-i, random_state=0)
    
        cl=discriminant_analysis.QuadraticDiscriminantAnalysis()                                                                    
        cl.fit(X_train,Y_train)
        Z=cl.predict(X_test)
        scores = metrics.accuracy_score(Y_test, Z)*100
        
        print("Quadratic Discriminant Analysis. Training:" , i*100 ,"%")
        print(metrics.classification_report(Y_test,Z))
        print(metrics.confusion_matrix(Y_test,Z))
        print("Accuracy: %0.2f" % (scores))
        
        myList.append(scores)
        
    return myList
        
    
def SGD(s,e):
    myList = []
    f=open("wdbc.data.txt")
    data = pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")
    X=np.array(data)
    Y=X[:,1]
    Y=np.where(Y=='M',1,0)
    X=X[:, s:e]
    
    seq = [.9, .8, .5, .25]
    for i in seq:
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=i, test_size=1-i, random_state=0)
    
        cl = linear_model.SGDClassifier(loss="perceptron", penalty="elasticnet", max_iter=600)                                                                 
        cl.fit(X_train,Y_train)
        Z=cl.predict(X_test)
        scores = metrics.accuracy_score(Y_test, Z)*100
        
        print("Stochastic Gradient Descent. Training:" , i*100 ,"%")
        print(metrics.classification_report(Y_test,Z))
        print(metrics.confusion_matrix(Y_test,Z))
        print("Accuracy: %0.2f" % (scores))
        
        myList.append(scores)
        
    return myList
    
    
#%%

def LogReg10():
    myList = []
    f=open("wdbc.data.txt")                                                     
    data= pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")                                                                                       
    X=np.array(data)                                                            
    Y=X[:,1]                                                                                                                                            
    Y= np.where(Y=='M', 1, 0)                                                   
    
    X_new =  X[:,[2,4,5,8,9,15,22,24,25,29]]   

    seq = [.9, .8, .5, .25]

    for i in seq:
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_new, Y, train_size=i, test_size= 1-i, random_state=0)
        
        cl = linear_model.LogisticRegression(C=1e5)
        cl.fit(X_train,Y_train)
        Z=cl.predict(X_test)
        scores = metrics.accuracy_score(Y_test, Z)*100
        
        print("Logistic Regression. Training:" , i*100 ,"%")
        print(metrics.classification_report(Y_test,Z))
        print(metrics.confusion_matrix(Y_test,Z))
        print("Accuracy: %0.2f" % (scores))
            
        myList.append(scores)
    
    print("\n")
    return myList
        

def SVM10():
    myList = []
    f=open("wdbc.data.txt")
    data= pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")
    X=np.array(data)
    Y=X[:,1]
    Y= np.where(Y=='M', 1, 0)
    
    X_new =  X[:,[2,4,5,8,9,15,22,24,25,29]]   

    seq = [.9, .8, .5, .25]
    for i in seq:
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_new, Y, train_size=i, test_size=1-i, random_state=0)
    
        cl=svm.SVC(kernel='linear')                                                           
        cl.fit(X_train,Y_train)
        Z=cl.predict(X_test)
        scores = metrics.accuracy_score(Y_test, Z)*100
        
        print("Support Vector. Training:" , i*100 ,"%")
        print(metrics.classification_report(Y_test,Z))
        print(metrics.confusion_matrix(Y_test,Z))
        print("Accuracy: %0.2f" % (scores))
        
        myList.append(scores)
    
    
        
    return myList

           
def DTC10():
    myList = []
    f=open("wdbc.data.txt")
    data= pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")
    X=np.array(data)
    Y=X[:,1]
    Y= np.where(Y=='M', 1, 0)

    X_new =  X[:,[2,4,5,8,9,15,22,24,25,29]]   
    
    seq = [.9, .8, .5, .25]
    for i in seq:
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_new, Y, train_size=i, test_size=1-i, random_state=0)
    
        cl = tree.DecisionTreeClassifier()                                                                
        cl.fit(X_train,Y_train)
        Z=cl.predict(X_test)
        scores = metrics.accuracy_score(Y_test, Z)*100
        
        print("Decision Tree. Training:" , i*100 ,"%")
        print(metrics.classification_report(Y_test,Z))
        print(metrics.confusion_matrix(Y_test,Z))
        print("Accuracy: %0.2f" % (scores))
        
        myList.append(scores)
    return myList


def KNC10():
    myList = []
    f=open("wdbc.data.txt")
    data = pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")
    X=np.array(data)
    Y=X[:,1]
    Y=np.where(Y=='M',1,0)
    
    X_new =  X[:,[2,4,5,8,9,15,22,24,25,29]] 
    
    seq = [.9, .8, .5, .25]
    for i in seq:
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_new, Y, train_size=i, test_size=1-i, random_state=0)
    
        cl = neighbors.KNeighborsClassifier(n_neighbors=3)                                                             
        cl.fit(X_train,Y_train)
        Z=cl.predict(X_test)
        scores = metrics.accuracy_score(Y_test, Z)*100
        
        print("K Neighbors. Training:" , i*100 ,"%")
        print(metrics.classification_report(Y_test,Z))
        print(metrics.confusion_matrix(Y_test,Z))
        print("Accuracy: %0.2f" % (scores))
        
        myList.append(scores)
        
    return myList


def RFC10():
    myList = []
    f=open("wdbc.data.txt")
    data=pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")
    X=np.array(data)
    Y=X[:,1]
    Y=np.where(Y=='M',1,0)
    
    X_new =  X[:,[2,4,5,8,9,15,22,24,25,29]] 
    
    seq = [.9, .8, .5, .25]
    for i in seq:
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_new, Y, train_size=i, test_size=1-i, random_state=0)
    
        cl=ensemble.RandomForestClassifier(max_depth=15, n_estimators=10, max_features=1)
        cl.fit(X_train,Y_train)
        Z=cl.predict(X_test)
        scores = metrics.accuracy_score(Y_test, Z)*100
        
        print("Random Forest. Training:" , i*100 ,"%")
        print(metrics.classification_report(Y_test,Z))
        print(metrics.confusion_matrix(Y_test,Z))
        print("Accuracy: %0.2f" % (scores))
        
        myList.append(scores)
        
    return myList
    
    
def MLP10():
    myList = []
    f=open("wdbc.data.txt")
    data = pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")
    X=np.array(data)
    Y=X[:,1]
    Y=np.where(Y=='M',1,0)

    X_new =  X[:,[2,4,5,8,9,15,22,24,25,29]] 
    
    seq = [.9, .8, .5, .25]
    for i in seq:
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_new, Y, train_size=i, test_size=1-i, random_state=0)
    
        cl = neural_network.MLPClassifier(activation='logistic', solver='lbfgs', max_iter=1000 )                                                                   
        cl.fit(X_train,Y_train)
        Z=cl.predict(X_test)
        scores = metrics.accuracy_score(Y_test, Z)*100
        
        print("Multi Layer Perceptron. Training:" , i*100 ,"%")
        print(metrics.classification_report(Y_test,Z))
        print(metrics.confusion_matrix(Y_test,Z))
        print("Accuracy: %0.2f" % (scores)) 
        
        myList.append(scores)
        
    return myList
    
def ABC10():
    myList = []
    f=open("wdbc.data.txt")
    data=pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")
    X=np.array(data)
    Y=X[:,1]
    Y=np.where(Y=='M',1,0)
    
    X_new =  X[:,[2,4,5,8,9,15,22,24,25,29]] 
    
    seq = [.9, .8, .5, .25]
    for i in seq:
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_new, Y, train_size=i, test_size=1-i, random_state=0)
    
        cl=ensemble.AdaBoostClassifier()                                                                    
        cl.fit(X_train,Y_train)
        Z=cl.predict(X_test)
        scores = metrics.accuracy_score(Y_test, Z)*100
        
        print("Ada Boost. Training:" , i*100 ,"%")
        print(metrics.classification_report(Y_test,Z))
        print(metrics.confusion_matrix(Y_test,Z))
        print("Accuracy: %0.2f" % (scores))
        
        myList.append(scores)
        
    return myList
        
    
def GNB10():
    myList = []
    f=open("wdbc.data.txt")
    data=pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")
    X=np.array(data)
    Y=X[:,1]
    Y=np.where(Y=='M',1,0)
    
    X_new =  X[:,[2,4,5,8,9,15,22,24,25,29]] 
    
    seq = [.9, .8, .5, .25]
    for i in seq:
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_new, Y, train_size=i, test_size=1-i, random_state=0)
    
        cl=naive_bayes.GaussianNB()
        cl.fit(X_train,Y_train)
        Z=cl.predict(X_test)
        scores = metrics.accuracy_score(Y_test, Z)*100
        
        print("Gaussian Naive Bayes. Training:" , i*100 ,"%")
        print(metrics.classification_report(Y_test,Z))
        print(metrics.confusion_matrix(Y_test,Z))
        print("Accuracy: %0.2f" % (scores))
        
        myList.append(scores)
    return myList
    
    
def QDA10():
    myList = []
    f=open("wdbc.data.txt")
    data=pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")
    X=np.array(data)
    Y=X[:,1]
    Y=np.where(Y=='M',1,0)
    
    X_new =  X[:,[2,4,5,8,9,15,22,24,25,29]] 
    
    seq = [.9, .8, .5, .25]
    for i in seq:
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_new, Y, train_size=i, test_size=1-i, random_state=0)
    
        cl=discriminant_analysis.QuadraticDiscriminantAnalysis()                                                                    
        cl.fit(X_train,Y_train)
        Z=cl.predict(X_test)
        scores = metrics.accuracy_score(Y_test, Z)*100
        
        print("Quadratic Discriminant Analysis. Training:" , i*100 ,"%")
        print(metrics.classification_report(Y_test,Z))
        print(metrics.confusion_matrix(Y_test,Z))
        print("Accuracy: %0.2f" % (scores))
        
        myList.append(scores)
        
    return myList
        
    
def SGD10():
    myList = []
    f=open("wdbc.data.txt")
    data = pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")
    X=np.array(data)
    Y=X[:,1]
    Y=np.where(Y=='M',1,0)
    
    X_new =  X[:,[2,4,5,8,9,15,22,24,25,29]] 

    seq = [.9, .8, .5, .25]
    for i in seq:
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_new, Y, train_size=i, test_size=1-i, random_state=0)
    
        cl = linear_model.SGDClassifier(loss="perceptron", penalty="elasticnet", max_iter=600)                                                                 
        cl.fit(X_train,Y_train)
        Z=cl.predict(X_test)
        scores = metrics.accuracy_score(Y_test, Z)*100
        
        print("Stochastic Gradient Descent. Training:" , i*100 ,"%")
        print(metrics.classification_report(Y_test,Z))
        print(metrics.confusion_matrix(Y_test,Z))
        print("Accuracy: %0.2f" % (scores))
        
        
        myList.append(scores)
        
    return myList
    

    