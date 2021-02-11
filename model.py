#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob


# In[2]:


# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'train'
valid_path = 'test'


# In[3]:


train_path


# In[4]:


# Import the Vgg 16 library as shown below and add preprocessing layer to the front of VGG
# Here we will be using imagenet weights


vgg16 = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[5]:


# don't train existing weights
for layer in vgg16.layers:
    layer.trainable = False


# In[6]:



# useful for getting number of output classes
folders = glob('train/*')


# In[7]:


folders


# In[8]:


# our layers - you can add more if you want
x = Flatten()(vgg16.output)


# In[9]:


prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg16.input, outputs=prediction)


# In[10]:


# view the structure of the model
model.summary()


# In[11]:


# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[12]:



# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[13]:


# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


# In[14]:


len(training_set)


# In[15]:


test_set = test_datagen.flow_from_directory('test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[16]:


# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


# In[17]:


import matplotlib.pyplot as plt


# In[18]:


plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[41]:


r.history['val_accuracy']


# In[19]:


# save it as a h5 file


from tensorflow.keras.models import load_model

model.save('vgg16.h5')


# In[20]:


y_pred = model.predict(test_set)


# In[21]:


y_pred


# In[22]:


import numpy as np
y_pred1 = np.argmax(y_pred, axis=1)


# In[23]:


y_pred1


# In[24]:


test_set.classes


# In[53]:


import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('C:/Users/anjit/Machine Learning/deep learning/Cotton Disease/test/fresh cotton plant/dsd (140)_iaip.jpg', target_size = (224,224))
test_image = image.img_to_array(test_image)
test_image=test_image/255
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)


# In[54]:


result


# In[55]:


a= np.argmax(model.predict(test_image),axis=1)


# In[56]:


a


# In[57]:


if a==0:
    print("diseased cotton leaf")
if a==1:
    print("diseased cotton plant")
if a==2:
    print("fresh cotton leaf")
if a==3:
    print("fresh cotton plant")


# In[ ]:




