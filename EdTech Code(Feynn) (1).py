#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


data = pd.read_csv("2015_16_Districtwise.csv")


# In[4]:


data.head()


# In[6]:


df_1=data.iloc[:,129:179]
df_1


# In[7]:


data.drop(columns=df_1,inplace=True)
print(data.shape)


# In[8]:


df_3=data.iloc[:,73:89]


# In[12]:


data.set_index('STATNAME',inplace = True)


# In[13]:


data.describe().transpose()


# In[16]:


data.drop(columns=['SCH9','SCH9GA','SCHBOY9','SCHGIR9','ENR9'],inplace=True)


# In[40]:


kmeans = KMeans(n_clusters=3)
label = kmeans.fit_predict(X1)
label


# In[41]:


print(kmeans.cluster_centers_)


# In[ ]:





# In[19]:


from matplotlib import pyplot as plt 


# In[20]:


X1 = data.loc[:,["SCHTOT","SCHTOTGA"]]
from sklearn.cluster  import KMeans
wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k,init = 'k-means++')
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(15,7))
plt.grid()
plt.plot(range(1,11),wcss,linewidth=2,color='red',marker="8")
plt.xlabel('k value')
plt.ylabel('WCSS')
plt.show()


# In[21]:


X1 = data.loc[:,["SELETOT","SCHTOTGA"]]
from sklearn.cluster  import KMeans
wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k,init = 'k-means++')
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(15,7))
plt.grid()
plt.plot(range(1,11),wcss,linewidth=2,color='blue',marker="8")
plt.xlabel('k value')
plt.ylabel('WCSS')
plt.show()


# In[22]:


import numpy as np


# In[28]:


Total=['SGTOILTOT','SCHBOYTOT','SWATTOT','SELETOT','SCOMPTOT']


# In[30]:


data.loc[['JAMMU & KASHMIR','HIMACHAL PRADESH','PUNJAB'],Total].plot()
plt.title('school Fcaility')
plt.xlabel('Total facilities')
plt.ylabel('states')


# In[38]:


data['SCHTOTGA'].plot(kind='hist',figsize=(8,5))
plt.title('Total Schools by Category-government aided')
plt.xlabel('Total number of School-government school')
plt.ylabel('Number of States')
plt.show


# In[39]:


data['SCHTOT'].plot(kind='hist',figsize=(8,5))
plt.title('Total Schools by Category')
plt.xlabel('Total number of schools')
plt.ylabel('Number of States')
plt.show


# In[ ]:




