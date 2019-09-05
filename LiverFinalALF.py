# In[ ]:
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
tf.reset_default_graph()


# In[20]:


f1 = sys.argv[1] 
f2 = sys.argv[2]
f3 = sys.argv[3]
f4 = sys.argv[4]
f5 = sys.argv[5]
f6 = sys.argv[6]
f7 = sys.argv[7]
f8 = sys.argv[8]
f9 = sys.argv[9]
f10 = sys.argv[10]
f11 = sys.argv[11]
f12 = sys.argv[12]
f13 = sys.argv[13]
f14 = sys.argv[14]
f15 = sys.argv[15]


f111 = int(f1)
f22 = int(f2)
f33 = float(f3)
f44 = float(f4)
f55 = float(f5)
f66 = float(f6)
f77 = float(f7)
f88 = float(f8)
f99 = float(f9)
f100 = float(f10)
f101 = float(f11)
f102 = float(f12)
f103 = float(f13)
f104 = float(f14)
f105 = float(f15)
f = [[f111,f22,f33,f44,f55,f66,f77,f88,f99,f100,f101,f102,f103,f104,f105]]


# In[ ]:


sc = StandardScaler()
pred = sc.fit_transform(f)

#data = [[25, 1,65, 150, 22.7, 0, 70.6, 110, 80, 50, 120, 170, 0,0, 0]]
td = pd.DataFrame(f, columns = ['Age','Gender', 'Weight', 'Height', 'Body Mass Index', 'Obesity', 'Waist', 'Maximum Blood Pressure', 'Minimum Blood Pressure', 'Good Cholesterol', 'Bad Cholesterol', 'Total Cholesterol', 'Dyslipidemia', 'Alcohol Consumption', 'HyperTension'])
td = sc.transform(td)
# In[2]:


#model.summary()


# In[9]:




# In[10]:


#t = np.array([0.61337158, -0.33789299, -0.40783724, -0.31345325, -0.33450163,-0.32189122,-0.24325881, -0.29278209,-0.45193161,1.625,-0.58561035])


# In[11]:


#t = np.array(pred)
#t.shape


# In[16]:


#s=t.reshape(1, -1)


# In[17]:
model = Sequential()
#model.add(Dense(100, input_shape=(10,)))
#model.add(Activation('relu'))

#model.add(Dense(10))
#model.add(Activation('softmax'))

model.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))
model.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# compile ANN
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.load_weights('C:/Users/Hatim/Desktop/liverALF3k_wts.h5')
# model = load_model('C:/Users/Hatim/Desktop/liverALF_wts.h5')

#s.shape


# In[18]:

y_pred = model.predict(td)
y_pred = [ 1 if y>0.9 else 0 for y in y_pred ]
print(y_pred)

# In[7]:


#model.predict([0.61337158, -0.33789299, -0.40783724, -0.31345325, -0.33450163,-0.32189122,-0.24325881, -0.29278209,-0.45193161,1.625,-0.58561035])


# In[ ]:





# In[ ]:





# In[ ]:









