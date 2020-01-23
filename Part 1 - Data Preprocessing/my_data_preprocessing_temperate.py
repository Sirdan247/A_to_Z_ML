#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data Preprocessing

# import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# import the dataset
dataset = pd.read_csv('Data.csv')
dataset.head()

# X = dataset.iloc[:,:-1]
# y = dataset.iloc[:, 3]



# Encoding categorical data
# Use only LabelEncoder for dependent vector (y) and for any ranked independent matrix (X)
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_X = LabelEncoder()
# X['Country'] = labelencoder_X.fit_transform(X['Country'])
# onehotencoder = OneHotEncoder(categorical_features = [0])
# X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [0])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)

X = pd.DataFrame(np.array(ct.fit_transform(X), dtype=np.int))

le_y = LabelEncoder()
y = pd.DataFrame(le_y.fit_transform(y), columns =['Purchased'])



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split # Used to be sklearn.cross_validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)




# Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[14]:


# Importing the dataset

dataset = pd.read_csv('Social_Network_Ads.csv')
dataset.head()

X = dataset.iloc[:,[2,3]] # To be in matrix form
y = dataset.iloc[:, 4]


# In[15]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split # Used to be sklearn.cross_validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[16]:


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[17]:


# Fitting the classifier



# In[18]:


# Predicting the test set results
y_pred = classifier.predict(X_test)


# In[19]:


y_pred


# In[20]:


# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[21]:


cm


# In[25]:


# Visualizing the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min()-1, stop= X_set[:, 0].max()+1, step = 0.01),
                   np.arange(start=X_set[:, 1].min()-1, stop= X_set[:, 1].max()+1, step = 0.01 ))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(y_set)):
             plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()


# In[27]:


# Visualizing the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min()-1, stop= X_set[:, 0].max()+1, step = 0.01),
                   np.arange(start=X_set[:, 1].min()-1, stop= X_set[:, 1].max()+1, step = 0.01 ))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(y_set)):
             plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

