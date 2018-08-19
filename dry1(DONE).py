
# coding: utf-8

# In[15]:


import pandas as pd
data = pd.read_csv('train.csv')
from sklearn import linear_model
data_x = pd.read_csv('test.csv')
data_y = pd.read_csv('sample_submission.csv')


# In[8]:


x_train = data[['Elevation']]
y_train = data[['Cover_Type']]
x_test = data_x[['Elevation']]
y_test = data_y[['Cover_Type']]


# In[9]:


linear = linear_model.LinearRegression()
linear.fit(x_train , y_train)
linear.score(x_train , y_train)


# In[10]:


print('Coefficient: \n' , linear.coef_)
print('Intercept: \n' , linear.intercept_)


# In[11]:


prediction = linear.predict(x_test)


# In[14]:


linear.score(x_test, y_test)


# In[ ]:


#НОЛЬ, ничего не предсказалось, но так и должно быть, было бы удивительно, если бы с одного столбца мы бы что-то получили
#далее можно идти по каждому столбцу так и комбинить, но лучше поанализировать данные и дропнуть лишние затем обучить на всей 
#отобранной выборке

