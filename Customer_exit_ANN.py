# Part 1: Preprocessing

# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:\Docs\DS\Customer_exit_ANN\BankCustomers.csv')

# Stroring 3rd column to 13th column in x
# First and second column are not important for the model
X = dataset.iloc[:, 3:13]
# Adding .value at the end will give an object as output, not a dataframe

# Last column is the customer exit. Stored separately in y
y = dataset.iloc[:, 13].values

# Convert categorical variables into dummy variables
states=pd.get_dummies(X['Geography'],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)

# Concatenating the dummy variables with the independent variables dataset  
X=pd.concat([X,states,gender],axis=1)

# Drop the geography and gender columns which are no longer required
X=X.drop(['Geography','Gender'],axis=1)

# Splitting the data into training set and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling using standard scaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2: Making the model

# Importing keras libraries and  packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing ANN
# Sequential makes it a sequential neural network
classifier = Sequential()

# Adding the input layer and the first hidden layer
# Relu is an activation function
# Dense creates the hidden layers
# There are eleven inputs corresponding to eleven columns in first layer and six outputs
# Initially all the weights are assigned randomy, which is called "uniform" in keras parlance 
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))

# Adding the second hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

# Adding the output layer
# The activation function is sigmoid
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

# Compiling the ANN
# Adam tries to reduce loss function
# the loss funstion is binary_crossaccuracy (because the output is binary), the metric is accuracy
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# Batch size is the number of forward propogation and back propogation
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# This gives the values between 0 and 1. 0 is not exit, 1 is exit
y_pred = (y_pred > 0.5)
# This gives true and false whether the predicted value above 0.5

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy=accuracy_score(y_test,y_pred)
# The accuracy is 85.65% for prediction
