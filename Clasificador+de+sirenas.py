
# coding: utf-8

# # Clasificación de sirenas (Endemicas y Migrantes)

# ## Importación de librerías

# In[4]:


import numpy as np
from sklearn import tree
import pandas as pd


# In[5]:


data = pd.read_csv('datasets/sirenas_endemicas_y_sirenas_migrantes_historico.csv')


# ## Visualización de datos
# Primero se vieron todos los datos, y lo que alcanzé a ver es que tal vez los datos de sirenas migrantes y sirenas endemicas estaban separando por la mitad el dataset

# In[101]:


data


# In[102]:


data.shape


# ### Análisis de la columna especies
# Se separó la columna para ver si el dataset estaba partido en dos

# In[103]:


sirenas = data.pop('especie')


# Se vió el número de datos por categoría y resutó ser 50% sirena endemicas y 50% sirenas migrantes

# In[104]:


sirenas.value_counts()


# Se separaron los datos a la mitad del data set y efectivamente los datos de sirenas migrantes estaban en la parte de arriba y los datos de sirenas endémicas estaban en la parte de abajo, lo que causaría problemas al elegir una parte del dataset para entrenamiento y otro para pruebas

# In[105]:


migrantes = sirenas[:50]
migrantes.shape


# In[106]:


migrantes.value_counts()


# In[107]:


endemicas = sirenas[50:100]
endemicas.shape


# In[108]:


endemicas.value_counts()


# ## Se regresa el dataset a su estado original

# In[109]:


data["especie"] = sirenas
data.shape


# Como ya sabemos que el dataset está partido en 2 vamos a separar las dos partes para poder tener la misma cantidad de datos de sirenas endemicas y sirenas migrantes para datos de entrenamiento y de prueba

# In[110]:


data_migrantes = data.iloc[:50]
data_migrantes.shape


# In[111]:


data_migrantes.head()


# In[112]:


data_endemicas = data.iloc[50:100]
data_endemicas.shape


# In[113]:


data_endemicas.head()


# ### Datos de entrenamiento
# Se seleccionaron el 80% de los datos para entrenar el modelo, 40% de sirenas endemicas y 40% de sirenas migrantes

# In[114]:


frames = [data_migrantes[:40], data_endemicas[:40]]
X = pd.concat(frames)
X.shape


# ### Datos de prueba
# Del el 20% de los datos restantes se seleccionaron 10% de sirenas migrantes y 10% de sirenas endemicas para probar el modelo

# In[115]:


frames = [data_migrantes[40:50], data_endemicas[40:50]]
X_test = pd.concat(frames)
X_test.shape


# In[116]:


X_test


# ### Cambiar datos categóricos
# La clasificación de datos en el dataset son datos categóricos, así que se reemplazaron con datos numéricos y fue guardada en las variables "y" y "y_test"

# In[117]:


y = pd.get_dummies(X.pop("especie"))


# In[118]:


y_test = pd.get_dummies(X_test.pop("especie"))


# In[119]:


y.shape


# In[120]:


y_test.shape


# In[121]:


X.shape


# In[122]:


X_test.shape


# In[123]:


y.head()


# ## Entrenamiento del modelo (árbol de decisiones)

# Se creó el modelo y se entrenó con un arbol de decisiones binario con el criterio gini

# In[124]:


model = tree.DecisionTreeClassifier()


# In[125]:


model.fit(X,y)


# ### Se obtuvo el error absoluto y la precisión (Creo que los dos están mal)

# In[126]:


from sklearn import metrics
metrics.mean_absolute_error(y_test, model.predict(X_test))


# In[127]:


from sklearn import cross_validation
resultado =  cross_validation.cross_val_score(model, X,y,cv=5,scoring='accuracy')
print(resultado)


# ## Análisis gráfico
# Ahora vemos por que en los modelos obtenemos una presición del 100% y un error absoluto de 0, pues si existe una gran diferencia entre todos los datos de las sirenas endémicas y migrantes

# In[6]:


import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.set(style="ticks")

sns.pairplot(data, hue="especie")


# ## Preparación de datos de predicción
# Se leyeron y analizaron los datos para predecir

# In[128]:


result = pd.read_csv('datasets/sirenas_endemicas_y_sirenas_migrantes.csv')


# In[129]:


result


# Se quitó la columna de especie para poder predecir el dataframe completo

# In[130]:


result.pop("especie")


# Se predijeron los datos

# In[131]:


result_predict = model.predict(result)
result_predict


# In[132]:


pd_result = pd.DataFrame(data=result_predict,
                        columns=['sirena_endemica','sirena_migrante'])
pd_result.head()


# Los datos resultado de la predicción no fueron categóricos, así que se convirtieron a categoricos y se agregaron al dataframe result

# In[133]:


result["especie"] = pd_result.idxmax(axis=1)


# In[134]:


result


# Por último se imprimió el archivo resultado_sirenas.csv con los datos de result.

# In[135]:


result.to_csv('datasets/resultado_sirenas.csv')

