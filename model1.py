#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 09:31:37 2019

@author: xander999
"""
# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data1.csv')
X = dataset.iloc[:, 2:18].values
y = dataset['letter-name']

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
'''
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
'''
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
yy=labelencoder_y.fit_transform(y)

from keras.utils import to_categorical
y = to_categorical(y)

#Convert back to numerical data
from numpy import argmax
ytret=[argmax(a) for a in y]
ytret=np.asarray(ytret)

#y=y/25

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu', input_dim = 16))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu'))

# Adding the third hidden layer
classifier.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 26, init = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# optimizer= rmsprop/adam/
# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 60)
classifier.fit(X_train, y_train, batch_size = 15, nb_epoch = 60)
classifier.fit(X_train, y_train, batch_size = 25, nb_epoch = 90)

#batch_size=10, nb_epoch=100
#batch_size=128, nb_epoch=20
# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) 


y_test=[argmax(a) for a in y_test]
y_test1=np.asarray(y_test)

y_pred1=[argmax(a) for a in y_pred]
y_pred1=np.asarray(y_pred1)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test1, y_pred1)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test1,y_pred1)
print(accuracy*100)
