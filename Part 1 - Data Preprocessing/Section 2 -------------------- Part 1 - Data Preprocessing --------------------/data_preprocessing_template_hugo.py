# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:11:55 2019

@author: htalmeida1
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')

# Creating the matrix of features (the 3 independent variables)

# Getting all the lines and all the columns except for the last one
X = dataset.iloc[:,:-1].values

#Creating the dependent variable vector
y = dataset.iloc[:, 3].values

# Dealing with the missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy = "mean")
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


# Dealing with the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
# the column becomes the array of encoded values
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# In order to prevent the model from thinking their numeric values are important,
# we should use dummy encoding.
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# To deal with the categorical dependent variable, we can just use LabelEncoder,
# because, since it is a dependent variable, and it is a classification problem
# the model already knows its values don't hold numeric significance.

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Spliting into train and test sets.
from sklearn.model_selection import train_test_split
 # random state to always get the same result
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling
# standardisation: xstand = (x-mean(x))/std(x)
# normalisation: xnorm = (x-min(x))/(max(x)-min(x))
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

# In the test set we just transform it according to what was fitted in the training set.
# In other words, we are applying the same transformation in both sets.
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# For classification problems we don't need to apply scaling, but for regression we do.
