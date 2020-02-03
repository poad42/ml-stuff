#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[25]:


#feature_set = np.random.rand(5,3) #generate a random array of values#feature set needs to be constant for replicable results
feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]]) 
labels = np.array([[1,0,0,1,1]])  
labels = labels.reshape(5,1) #reshape as a vector


# In[26]:


np.random.seed(42)  
weights = np.random.rand(3,1) #set weights  
bias = np.random.rand(1)  #set bias
lr = 0.05 #set learning rate


# In[27]:


def sigmoid(x):  
    return 1/(1+np.exp(-x)) #define the sigmoid fuction, switch to RELU later


# In[28]:


def sigmoid_der(x):  
    return sigmoid(x)*(1-sigmoid(x)) #calculate the derivative


# In[32]:


for epoch in range(20000):  
    inputs = feature_set

    # feedforward step1
    XW = np.dot(feature_set, weights) + bias

    #feedforward step2
    z = sigmoid(XW)


    # backpropagation step 1
    error = z - labels

    print("Epoch:",epoch,",Error:",error.sum())

    # backpropagation step 2
    dcost_dpred = error
    dpred_dz = sigmoid_der(z)

    z_delta = dcost_dpred * dpred_dz

    inputs = feature_set.T
    weights -= lr * np.dot(inputs, z_delta)

    for num in z_delta:
        bias -= lr * num


# In[34]:


single_point = np.array([1,0,0]) #vary linear inputs to get diffrent predictions of the output y 
result = sigmoid(np.dot(single_point, weights) + bias)  
print(result) 

