
# coding: utf-8

# In[1]:


import numpy as np
from sklearn import tree
import pandas as pd


# In[2]:


data = pd.read_csv('datasets/sirenas_endemicas_y_sirenas_migrantes_historico.csv')


# In[3]:


data.head(100)


# In[4]:


y = pd.get_dummies(data.pop("especie"))


# In[5]:


y.shape


# In[6]:


data.shape


# In[7]:


y.head()


# In[8]:


model = tree.DecisionTreeClassifier()


# In[9]:


model.fit(data,y)


# In[10]:


result = pd.read_csv('datasets/sirenas_endemicas_y_sirenas_migrantes.csv')


# In[11]:


result


# In[12]:


result.pop("especie")


# In[13]:


result_predict = model.predict(result)
result_predict


# In[14]:


pd_result = pd.DataFrame(data=result_predict,
                        columns=['sirena_endemica','sirena_migrante'])
pd_result.head()


# In[15]:


result["especie"] = pd_result.idxmax(axis=1)


# In[16]:


result


# In[17]:


result.to_csv('datasets/resultado_sirenas.csv')

