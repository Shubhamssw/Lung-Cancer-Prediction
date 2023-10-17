#!/usr/bin/env python
# coding: utf-8

# 
# # Lung Cancer Prediction

# In[3]:


import numpy as np 


# In[4]:


import os
for dirname, _, filenames in os.walk("D:\Dataset\Lung Cancer\cancer patient data sets.csv"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[5]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


df=pd.read_csv('D:\Dataset\Lung Cancer\cancer patient data sets.csv')


# In[7]:


df.head()


# In[8]:


df.tail()


# In[9]:


df.info()


# In[10]:


df.describe()


# In[11]:


df.isnull().sum()


# In[12]:


df.columns


# In[13]:


df=df[['Age', 'Gender', 'Air Pollution', 'Alcohol use',
       'Dust Allergy', 'OccuPational Hazards', 'Genetic Risk',
       'chronic Lung Disease', 'Balanced Diet', 'Obesity', 'Smoking',
       'Passive Smoker', 'Chest Pain', 'Coughing of Blood', 'Fatigue',
       'Weight Loss', 'Shortness of Breath', 'Wheezing',
       'Swallowing Difficulty', 'Clubbing of Finger Nails', 'Frequent Cold',
       'Dry Cough', 'Snoring', 'Level']]


# In[14]:


df.head()


# In[15]:


df['Level'].value_counts()


# In[16]:


df=df.replace({'Level':{'Low': 1, 'Medium': 2, 'High': 3}})


# In[17]:


df.head()


# In[18]:


import seaborn as sns
sns.set()


# In[19]:


plt.figure(figsize = (18,9))
sns.heatmap(df.corr(), cmap='GnBu', annot=True)
plt.show()


# In[20]:


df['Smoking'].corr(df['Passive Smoker'])


# In[21]:


sns.heatmap(df.corr()[['Level']].sort_values(by='Level', ascending=False), vmin=-1, vmax=1, annot=True, cmap='GnBu')


# In[22]:


df.columns


# In[23]:


df=df[['Age','Coughing of Blood','Dust Allergy','Passive Smoker','OccuPational Hazards','Air Pollution','chronic Lung Disease','Shortness of Breath','Dry Cough','Snoring','Swallowing Difficulty','Level']]


# In[24]:


df.head()


# In[25]:


df['Level'].value_counts()


# In[26]:


X=df.drop('Level',axis=1)

y=df['Level']


# In[27]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=1)


# In[28]:


from sklearn.linear_model import LogisticRegression
Classifier = LogisticRegression(solver='liblinear')
Classifier.fit(X_train,y_train)


# In[29]:


y_test_hat = Classifier.predict(X_test)


# In[30]:


Results = pd.DataFrame({'Actual':y_test,'Predictions':y_test_hat})
Results.head(10)


# In[31]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_test_hat))


# In[32]:


y_train_hat = Classifier.predict(X_train)

print(accuracy_score(y_train,y_train_hat))


# In[33]:


y_test_hat_proba = Classifier.predict_proba(X_test)

print(y_test_hat_proba.shape)


# In[34]:


y_test_hat_proba[0:5,:]


# In[35]:


array_in_scientific = y_test_hat_proba[0:5,:]

# Convert to normal number format
array_in_normal = np.vectorize(lambda x: format(x, '.16f'))(array_in_scientific)

print(array_in_normal)


# In[36]:


pls = y_test_hat_proba[:,1]

Results = pd.DataFrame({'Actual':y_test,'Predictions':y_test_hat,'Prob(Class = 3)':pls})

Results.head(5)


# In[37]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_test_hat)

print(cm)


# In[38]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.title('Confusion Matrix - Test Data')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')


# In[39]:


from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report


# In[40]:


def perform(y_pred):
    print("Precision : ", precision_score(y_test, y_test_hat, average = 'micro'))
    print("Recall : ", recall_score(y_test, y_test_hat, average = 'micro'))
    print("Accuracy : ", accuracy_score(y_test, y_test_hat))
    print("F1 Score : ", f1_score(y_test, y_test_hat, average = 'micro'))
    cm = confusion_matrix(y_test, y_pred)
    print("\n", cm)
    print("\n")
    print("**"*27 + "\n" + " "* 16 + "Classification Report\n" + "**"*27)
    print(classification_report(y_test, y_test_hat))
    print("**"*27+"\n")
    
    cm = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=['Low', 'Medium', 'High'])
    cm.plot()


# In[41]:


perform(y_test_hat)


# In[42]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_test_hat))


# In[ ]:




