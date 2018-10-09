# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 15:07:56 2018

@author: Miriam
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
import os
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  
from sklearn import linear_model, datasets, metrics, svm, tree, neural_network, model_selection
from sklearn import neighbors, gaussian_process, ensemble, naive_bayes, discriminant_analysis
from scipy import stats


def Plot():
    f=open("wdbc.data.txt")
    data=pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")
    X=np.array(data)
    Y=X[:,1]
    Y=np.where(Y=='M',1,0)
    X=X[:,2:32]
    
    plt.figure(1)
    cm_bright = ListedColormap(["green", "red"])
    
    ax=plt.subplot()
    orig=ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright, edgecolors='k')
    ax.set_title('Dataset points')
    ax.legend([orig], ['DATASET POINTS'])
   
    plt.figure(2, figsize=(15,4))
    cm_bright = ListedColormap(["green", "red"])
    ax2=plt.subplot(221)
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=.9, test_size=.1, random_state=0) 
    train= ax2.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=cm_bright, edgecolors='k')
    test= ax2.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, alpha=0.6, cmap=cm_bright, edgecolors='k', marker='+')
    ax2.set_title('Training points=90%, Test points=10%')
    ax2.legend([train, test],['TRAINING POINTS', 'TEST POINTS'])
    
    ax3=plt.subplot(222)
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=.8, test_size=.2, random_state=0)
    train= ax3.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=cm_bright, edgecolors='k')
    test= ax3.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, alpha=0.6, cmap=cm_bright, edgecolors='k', marker='+')
    ax3.set_title('Training points=80%, Test points=20%')
    ax3.legend([train, test],['TRAINING POINTS', 'TEST POINTS'])
    
    ax4=plt.subplot(223)
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=.5, test_size=.5, random_state=0)
    train= ax4.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=cm_bright, edgecolors='k')
    test= ax4.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, alpha=0.6, cmap=cm_bright, edgecolors='k', marker='+')
    ax4.set_title('Training points=50%, Test points=50%')
    ax4.legend([train, test],['TRAINING POINTS', 'TEST POINTS'])
    
    ax5=plt.subplot(224)
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=.25, test_size=.75, random_state=0)
    train= ax5.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=cm_bright, edgecolors='k')
    test= ax5.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, alpha=0.6, cmap=cm_bright, edgecolors='k', marker='+')
    ax5.set_title('Training points=25%, Test points=75%')
    ax5.legend([train, test],['TRAINING POINTS', 'TEST POINTS'])
        
    plt.show()
    
def Histo():
    f=open("wdbc.data.txt")                                                     
    data= pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")                                                                                       
    X=np.array(data)                                                                                                  
    
    s_coeff=[]
    x_bin=[]
    for i in range(2,32):
        sp_coeff = stats.spearmanr(X[:,i:i+1], X[:,1])
        print(X[:,i:i+1].size)
        print("feature: ", i , " ", sp_coeff)
        
        s_coeff.append(sp_coeff[0])
        x_bin.append(i)

    plt.figure(4)
    ax=plt.subplot()    
    
    plot=ax.hist(x_bin, weights=s_coeff, bins=30, alpha=0.75)
        
    ax.set_xlabel('Spearman Coefficients')
    ax.set_ylabel('Counts')
    ax.set_title('Histogram')
    ax.grid(axis='y')
    plt.show()


def SVCPlot(s,e):
    f=open("wdbc.data.txt")                                                     
    data= pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")                                                                                       
    X=np.array(data)                                                            
    Y=X[:,1]                                                                                                                                            
    Y= np.where(Y=='M', 1, 0)                                                   
    X=X[:,s:e] 

    
    if e-s == 2:   
        d=e-1 
        if e!=5 and e!=15 and e!=25 and d!=5 and d!=15 and d!=25 and s!=5 and s!=15 and s!=25:
            h = .02  # step size in the mesh
    
            # we create an instance of SVM and fit out data. We do not scale our
            # data since we want to plot the support vectors
            C = 1.0  # SVM regularization parameter
            svc = svm.SVC(kernel='linear', C=C).fit(X, Y)
            rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, Y)
            rbf_NuSVC = svm.NuSVC(kernel='rbf').fit(X, Y)
            lin_svc = svm.LinearSVC(C=C).fit(X, Y)
            
            # create a mesh to plot in
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            
            # title for the plots
            titles = ['SVC with linear kernel',
                      'LinearSVC (linear kernel)',
                      'SVC with RBF kernel',
                      'NuSVC with RBF kernel']
            
            
            for i, clf in enumerate((svc, lin_svc, rbf_svc, rbf_NuSVC)):
                # Plot the decision boundary. For that, we will assign a color to each
                # point in the mesh [x_min, m_max]x[y_min, y_max].
                plt.figure(3)
                plt.subplot(2, 2, i+1)
                plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                
                # Put the result into a color plot
                Z = Z.reshape(xx.shape)
                plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
            
                # Plot also the training points
                plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors='k')
                plt.xlim(xx.min(), xx.max())
                plt.ylim(yy.min(), yy.max())
                plt.xticks(())
                plt.yticks(())
                plt.title(titles[i])
            
            plt.show()
        else:
            print("I can't")
    
    else:
        print("I can't do a plot because I need ONLY 2 features! ")


def DTCPlot(s,e):
    print("Warning: old trees could be overwritten!")
    
    f=open("wdbc.data.txt")                                                     
    data= pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")                                                                                       
    X=np.array(data)                                                            
    Y=X[:,1]                                                                                                                                            
    Y= np.where(Y=='M', 1, 0)                                                   
    X=X[:,s:e] 
    
    seq=[.9, .8, .5, .25]
    for i, j in zip(seq, range(1, 10000)):
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=i, test_size=1-i, random_state=0)
        cl = tree.DecisionTreeClassifier()                                                                
        cl.fit(X_train,Y_train)
        Z=cl.predict(X_test)

        dot_data = tree.export_graphviz(cl, out_file=None) 
        graph = graphviz.Source(dot_data) 
        graph.render("wdbc_tree") 
        path='/Users/miria/OneDrive/Desktop/Primo_anno_LM/Progetto_Analisi_Dati/Breast-Cancer-Tumor-Classification--master'
        os.path.abspath(path)
        os.rename(os.path.join(path,'wdbc_tree.pdf'), os.path.join(path,'wdbc_tree'+str(j)+'.pdf'))

def Plot3B():
    data= pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")

    fig = plt.figure(5)
    
    ax = fig.add_subplot(111, projection='3d')
    
    f25=np.array(data[25])
    f24=np.array(data[24])
    f22=np.array(data[22])
    classification=np.array(data[1])
    
    #a1=np.column_stack((classification,radius))
    #a2=np.column_stack((classification,texture))
    
    malignant = np.where(classification == "M" )
    benign = np.where(classification=="B")
    ax.scatter(f25[malignant], f24[malignant], f22[malignant], marker='x', c='r')
    ax.scatter(f25[benign],f24[benign], f22[benign], marker='o', c='g')
    ax.set_xlabel('feature 25')
    ax.set_ylabel('feature 24')
    ax.set_zlabel('feature 22')
    
    plt.legend(['MALIGNANT', 'BENIGN'])
    plt.show()

