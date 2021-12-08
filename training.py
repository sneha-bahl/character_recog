
# coding: utf-8

# In[1]:


import numpy as np
from keras.applications.vgg16 import VGG16
import keras
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

get_ipython().magic(u'matplotlib inline')


# In[2]:


data = np.load("./data.npy")
labels = np.load("./labels.npy")

data,labels = shuffle(data,labels, random_state=0)

print (data.shape)
print (labels.shape)

print(labels[10],labels[100])


# In[3]:


labels = to_categorical(labels)
print (labels.shape)
print(labels)


# In[14]:


filters = 32
kernel_size = (3,3)
batch_size = 64
nb_epoch = 30


# In[15]:


'''
conv_model = Sequential()

conv_model.add(Convolution2D(filters, kernel_size[0], kernel_size[1], input_shape=(50, 50, 3)))
conv_model.add(Activation('relu'))
conv_model.add(Convolution2D(filters, kernel_size[0], kernel_size[1]))
conv_model.add(Activation('relu'))
conv_model.add(MaxPooling2D(pool_size=(2,2)))
conv_model.add(Dropout(0.25))
conv_model.add(Flatten())
conv_model.add(Dense(128))
conv_model.add(Activation('relu'))
conv_model.add(Dropout(0.5))
conv_model.add(Dense(62))
conv_model.add(Activation('softmax'))

conv_model.summary()



'''
model = Sequential()
model.add(Conv2D(64,(3,3), activation='relu', input_shape=(50,50,1)))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3), activation='relu'))
model.add(Conv2D(128,(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(256,(3,3), activation='relu'))
model.add(Conv2D(256,(3,3), activation='relu'))
model.add(Conv2D(256,(3,3), activation='relu'))
model.add(Conv2D(256,(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(512,(3,3), activation='relu'))
model.add(Conv2D(512,(3,3), activation='relu'))
model.add(Conv2D(512,(3,3), activation='relu'))
model.add(Conv2D(512,(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(512,(3,3), activation='relu'))
model.add(Conv2D(512,(3,3), activation='relu'))
model.add(Conv2D(512,(3,3), activation='relu'))
model.add(Conv2D(512,(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(62))
model.add(Activation('softmax'))

adam = Adam(lr=0.0003)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()


# In[16]:



model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])


# In[17]:


split = int(0.80 * data.shape[0])
x_train = data[:split, :, :, :]
x_test = data[split:, :, :, :]

y_train = labels[:split]
y_test = labels[split:]

print x_train.shape, y_train.shape
print x_test.shape, y_test.shape

'''
x_data = (data-data.mean(axis=0))/(data.std(axis=0) + 1e-8)
x_test = (x_test-x_test.mean(axis=0))/(x_test.std(axis=0) + 1e-8)
y_test = to_categorical(y_test)

'''

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# In[ ]:





# In[18]:


model.fit(x_train,y_train, shuffle=True, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1,validation_data=(x_test, y_test))


#hist =conv_model.fit(x_train, y_train, shuffle=True, batch_size=64, epochs=15, validation_data=(x_test, y_test))


# In[19]:



model.save('/home/sneha/Desktop/project/training_model/try.h5')  # creates a HDF5 file 'my_model.h5'
#f = h5py.File('/home/sneha/Desktop/project/training_model/try.h5', 'r+')
#del f['optimizer_weights']
#f.close()


# In[26]:


conv_model = Sequential()

conv_model.add(Conv2D(filters, kernel_size[0], kernel_size[1], input_shape=(50, 50, 1)))
conv_model.add(Activation('relu'))
conv_model.add(Conv2D(filters, kernel_size[0], kernel_size[1]))
conv_model.add(Activation('relu'))
conv_model.add(MaxPool2D(pool_size=(2,2)))
conv_model.add(Dropout(0.25))
conv_model.add(Flatten())
conv_model.add(Dense(128))
conv_model.add(Activation('relu'))
conv_model.add(Dropout(0.5))
conv_model.add(Dense(62))
conv_model.add(Activation('softmax'))

conv_model.summary()


# In[27]:


conv_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])


# In[28]:


conv_model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(x_test, y_test))


# In[ ]:




