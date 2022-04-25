#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

import sklearn.model_selection

import sklearn.tree
import plotly.express as px


# In[2]:


exo_train_df = pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/kepler-exoplanets-dataset/exoTrain.csv')
exo_train_df.head()


# In[3]:


exo_test_df = pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/kepler-exoplanets-dataset/exoTest.csv')
exo_test_df.head()


# In[4]:


exo_train_df.shape


# In[5]:


exo_test_df.shape


# In[6]:


exo_train_df.isnull().sum()


# In[7]:


exo_train_df.dropna()


# In[8]:


x_train = exo_train_df . iloc[:, 1 :]
x_train . head()


# In[9]:


y_train = exo_train_df . iloc[:, 0 ]
y_train . head()


# In[10]:


rf_clf = RandomForestClassifier(n_jobs =-1 , n_estimators =50 )
rf_clf.fit(x_train,y_train)


# In[11]:


rf_clf . score(x_train, y_train)


# In[12]:


x_test = exo_test_df . iloc[:, 1 :]
x_test . head()


# In[13]:


y_test = exo_test_df . iloc[:, 0 ]
y_test . head()


# In[14]:


y_predicted = rf_clf . predict(x_test)
y_predicted


# In[15]:


y_predicted = pd . Series(y_predicted)
y_predicted . head()


# In[16]:


confusion_matrix(y_test, y_predicted)


# In[17]:


print (classification_report(y_test, y_predicted))


# In[18]:


exo_train_df.describe()


# In[19]:


def mean_normalise(series):
    norm_series = (series - series.mean()) / (series.max() - series.min()) 
    return norm_series
  
  


# In[20]:


norm_train_df = exo_train_df.iloc[:, 1:].apply(mean_normalise, axis=1)
norm_train_df.head()


# In[21]:


norm_train_df.insert(loc=0, column='LABEL', value=exo_train_df['LABEL'])
norm_train_df.head()


# In[22]:


norm_test_df = exo_test_df.iloc[:,1:].apply(mean_normalise, axis=1)
norm_test_df.head()


# In[23]:


norm_test_df.insert(loc=0, column='LABEL', value=exo_test_df['LABEL'])
norm_test_df.head()


# In[24]:


def fast_fourier_transform(star):
    fft_star = np.fft.fft(star, n=len(star))
    return np.abs(fft_star)


# In[25]:


freq = np.fft.fftfreq(len(exo_train_df.iloc[0, 1:]))
freq


# In[26]:


x_fft_train_T = norm_train_df.iloc[:, 1:].T.apply(fast_fourier_transform)
x_fft_train = x_fft_train_T.T
x_fft_train.head()


# In[27]:


x_fft_test_T = norm_test_df.iloc[:, 1:].T.apply(fast_fourier_transform)
x_fft_test = x_fft_test_T.T
x_fft_test.head()


# In[28]:


y_train = norm_train_df['LABEL']
y_test = norm_test_df['LABEL']


# In[29]:


rf_clf = RandomForestClassifier(n_jobs=-1, n_estimators=50)
rf_clf.fit(x_fft_train, y_train)
print(rf_clf.score(x_fft_train, y_train))
y_pred = rf_clf.predict(x_fft_test)
y_pred


# In[30]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[31]:


print(classification_report(y_test, y_pred))


# In[32]:


model = sklearn.neighbors.KNeighborsClassifier()
model.fit(x_train, y_train);


# In[33]:


y_predicted = model.predict(x_test)
accuracy = sklearn.metrics.accuracy_score(y_test, y_predicted)
accuracy


# In[ ]:





# In[ ]:





# In[ ]:





# In[34]:


#from matplotlib import pyplot as plt
from sklearn import tree

model = sklearn.tree.DecisionTreeClassifier(max_depth=3)
model.fit(x_train, y_train);

#fig = plt.figure(figsize=(50,20))
tree.plot_tree(model);


# In[35]:


text_representation = tree.export_text(model)
print(text_representation)


# In[36]:


import graphviz

dot_data = tree.export_graphviz(model, out_file=None, 
                                #feature_names=iris.feature_names,  
                                #class_names=iris.target_names,
                                filled=True)

graph = graphviz.Source(dot_data, format="png") 
graph


# In[37]:


import graphviz

dot_data = tree.export_graphviz(model, out_file=None, filled=True)

graph = graphviz.Source(dot_data, format="png") 
graph


# # Feature engg
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[38]:


model = sklearn.svm.SVC()
model.fit(x_train, y_train);


# In[39]:


y_predicted = model.predict(x_test)
accuracy = sklearn.metrics.accuracy_score(y_test, y_predicted)
accuracy


# In[40]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[41]:


print(classification_report(y_test, y_pred))


# In[ ]:





# In[42]:


model = sklearn.neighbors.KNeighborsClassifier()
model.fit(x_fft_train, y_train);


# In[43]:


y_predicted = model.predict(x_fft_test)
accuracy = sklearn.metrics.accuracy_score(y_test, y_predicted)
accuracy


# In[44]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[45]:


print(classification_report(y_test, y_pred))


# In[56]:



x_fft_train = x_fft_train_T.T
x_fft_train.head()


# In[57]:


model = sklearn.neighbors.KNeighborsClassifier()
model.fit(x_fft_train, y_train);


# In[58]:


y_predicted = model.predict(x_fft_test)
accuracy = sklearn.metrics.accuracy_score(y_test, y_predicted)
accuracy


# In[59]:


print(classification_report(y_test, y_pred))


# In[ ]:


how to do smote???
 get_ipython().set_next_input(' smote not working');get_ipython().run_line_magic('pinfo2', 'working')
     get_ipython().set_next_input('     how to do xgboost');get_ipython().run_line_magic('pinfo', 'xgboost')
        get_ipython().set_next_input('        xgboost not working');get_ipython().run_line_magic('pinfo', 'working')
        

