
# coding: utf-8

# In[2]:


import os


# In[3]:


os.chdir("D:\\Data science\\Deep_Learning_A_Z\\Deep_Learning_A_Z\\Volume 1 - Supervised Deep Learning\\Part 1 - Artificial Neural Networks (ANN)\\Section 4 - Building an ANN")


# In[4]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[5]:


# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# In[6]:


#encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


# In[8]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[10]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[12]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[13]:


#initializing ANN
classifier = Sequential()


# In[14]:


#adding the input layer and first hidden layer
classifier.add(Dense(output_dim =6, init = "uniform", activation = 'relu', input_dim =11))


# In[15]:


#adding second hidden layer
classifier.add(Dense(output_dim =6, init = "uniform", activation = 'relu'))


# In[16]:


#adding output layer
classifier.add(Dense(output_dim =1, init = "uniform", activation = 'sigmoid'))


# In[17]:


#compiling ANN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])


# In[18]:


#fit ANN to training
classifier.fit(X_train, y_train, batch_size =10, nb_epoch =100)


# In[19]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[20]:


cm


# In[21]:


(1540+133)/2000


# In[26]:


## Predicting a single new observation
##"""Predict if the customer with the following informations will leave the bank:
#Geography: France
#Credit Score: 600
#Gender: Male
#Age: 40
#Tenure: 3
#Balance: 60000
#Number of Products: 2
#Has Credit Card: Yes
#Is Active Member: Yes
#Estimated Salary: 50000""###
new_predict = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_predict = (new_predict>0.5)


# In[27]:


new_predict


# In[ ]:


#k-fold cross validation
#Takes time to execute....***be cautious***....
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim =6, init = "uniform", activation = 'relu', input_dim =11))
    classifier.add(Dense(output_dim =6, init = "uniform", activation = 'relu'))
    classifier.add(Dense(output_dim =1, init = "uniform", activation = 'sigmoid'))
    classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch =50)
accuracies = cross_val_score(estimator = classifier, X= X_train, y =y_train, cv =10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()
    
    


# In[ ]:


#Parameter tuning  ***caution...it takes time to execute ******several hours*****

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim =6, init = "uniform", activation = 'relu', input_dim =11))
    classifier.add(Dense(output_dim =6, init = "uniform", activation = 'relu'))
    classifier.add(Dense(output_dim =1, init = "uniform", activation = 'sigmoid'))
    classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)


# In[ ]:


#.....caution .....execution may take....****several hours**** as my laptop does not have GPU****
Parameters = {"batch_size" : [25, 32],
              "nb_epoch" : [100, 200],
              "optimizer": ['adam', 'rmsprop']}
gridsearch = (estimator = classifier,
              param_grid = parameters,
              scoring = 'accuracy',
              CV = 10)
gridsearch = gridsearch.fit(X_train, y_train)
best_parameters = gridsearch.best_params_
best_accuracy = gridsearch.best_score_

