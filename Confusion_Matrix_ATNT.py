# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 18:13:40 2018

@author: heera
"""

import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.cluster import KMeans
from numpy import genfromtxt
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics import confusion_matrix

train_data=[]
test_data=[]
TrainX=[]
TrainY=[]
TestX=[]
TestY=[]

def pick_data(classes):
    my_data = genfromtxt('C:\\Datamining project1\\ATNTFaceImages400.csv', delimiter=',')
    a = np.transpose(my_data)
    b = np.int_(a)
    data = []
    for i in range(len(b)):
        for j in classes:
            if b[i][0] == j:
                data.append(b[i])
        np.savetxt('selected_data.csv', data, fmt='%d', delimiter=",")   
        
def test_train_data(training_instances,test_instances):    
    from numpy import genfromtxt
    number_of_classes = 10
    data = genfromtxt('selected_data.csv', delimiter=',')
    data = np.int_(data)
    array = np.roll(data,-1,-1)
    n=len(classes)   
    for x in range(0,n):            
            for y in range(0,training_instances):
                temp=y+(x*number_of_classes)
                train_data.append(array[temp])
            for z in range(training_instances,training_instances+test_instances):
                temp2=z+(x*number_of_classes)
                test_data.append(array[temp2])                
    np.savetxt('train_data.csv', train_data, fmt='%d', delimiter=",")
    np.savetxt('test_data.csv', test_data, fmt='%d', delimiter=",")
    for i in range (0,len(train_data)):
            TrainX.append(train_data[i][0:644])
            TrainY.append(train_data[i][644])
    for i in range (0,len(test_data)):
            TestX.append(test_data[i][0:644])
            TestY.append(test_data[i][644])     
    np.savetxt('TrainX.csv', TrainX, fmt='%d', delimiter=",") 
    np.savetxt('TrainY.csv', TrainY, fmt='%d', delimiter=",")
    np.savetxt('TestX.csv', TestX, fmt='%d', delimiter=",")
    np.savetxt('TestY.csv', TestY, fmt='%d', delimiter=",") 
    return TrainX, TrainY, TestX, TestY

def test_train_data_whole(training_instances,test_instances):    
    from numpy import genfromtxt
    number_of_classes = 10
    data = genfromtxt('selected_data.csv', delimiter=',')
    data = np.int_(data)
    array = np.roll(data,-1,-1)
    n=len(clas)   
    for x in range(0,n):            
            for y in range(0,training_instances):
                temp=y+(x*number_of_classes)
                train_data.append(array[temp])
            for z in range(training_instances,training_instances+test_instances):
                temp2=z+(x*number_of_classes)
                test_data.append(array[temp2])                
    np.savetxt('train_data.csv', train_data, fmt='%d', delimiter=",")
    np.savetxt('test_data.csv', test_data, fmt='%d', delimiter=",")
    for i in range (0,len(train_data)):
            TrainX.append(train_data[i][0:644])
            TrainY.append(train_data[i][644])
    for i in range (0,len(test_data)):
            TestX.append(test_data[i][0:644])
            TestY.append(test_data[i][644])     
    np.savetxt('TrainX.csv', TrainX, fmt='%d', delimiter=",") 
    np.savetxt('TrainY.csv', TrainY, fmt='%d', delimiter=",")
    np.savetxt('TestX.csv', TestX, fmt='%d', delimiter=",")
    np.savetxt('TestY.csv', TestY, fmt='%d', delimiter=",") 
    return TrainX, TrainY, TestX, TestY

#TrainX - Training Features
#TestX - Testing Features
#TrainY - Training Labels
#TestY - Test Labels

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) +1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    print ("Accuracy: ", end = '')
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    
lab = 0
val = int(input("Enter 0 to specify number of classes else press 1 to use the whole data: "))
if(val == 0):    
    num = int(input("Enter the number of classes that you want to consider: "))
    lab = num
    classes = []
    for i in range(num):
        j = int(input("Enter the "+ str(i+1) + " class you want to consider: "))
        classes.append(j)
    pick_data(classes) 
    training_instances=int(input("Enter the number of training elements: "))
    test_instances=10-training_instances    
    TrainX, TrainY, TestX, TestY = test_train_data(training_instances,test_instances) 
if(val == 1):
    lab = 40
    clas = []
    for i in range(40):
        clas.append(i+1)
    pick_data(clas)
    training_instances=int(input("Enter the number of training elements: "))
    test_instances=10-training_instances    
    TrainX, TrainY, TestX, TestY = test_train_data_whole(training_instances,test_instances) 
    

k = int(input("Enter a value for k: "))
kmeans = KMeans(n_clusters = k)
kmeans.fit(TrainX)
y_kmeans = kmeans.predict(TestX)
y_true = np.array(TestY, dtype=np.int64)
y_pred = []
y_pred = y_kmeans
d = confusion_matrix(y_true, y_pred)
h = d[1:]
df = pd.DataFrame(h)
s = np.delete(h,len(df.columns)-1,axis = 1)
print ("Confusion Matrix: ")
print (s)
print ("Column-Index: " )
ind = linear_assignment(s.max() - s)
print (ind)
print (cluster_acc(y_true, y_pred))
print ("Labels and Cluster Index")
print (y_true, y_pred)

















