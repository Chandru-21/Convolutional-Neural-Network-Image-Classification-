# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 11:06:08 2020

@author: Chandra mouli
"""

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

tf.__version__

#data preprocessing
#preprocessing the train    
#to avoid overfitting we apply transformations to the training sets like geometrical transfo like changing geometry,some rotations,zoom in and zoom out,shifiting pixels
#its also called image augmentation-transforming so that train set does not overtrains(overfitting)
train_datagen = ImageDataGenerator(
        rescale=1./255,#feature scaling to pixels by dividing values by 255 becoz each pixel values will be from 0 to 255 by divid we will get from 0 to 1
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)#to avoid overfitting


train_set = train_datagen.flow_from_directory(
        'C:\\Users\\Chandramouli\\.spyder-py3\\CNN\\dataset\\training_set',
        target_size=(64, 64),#final sizes of images fed to convo nn
        batch_size=32,
        class_mode='binary')#binary classification

#preprocessing the testset
test_datagen = ImageDataGenerator(rescale=1./255)#no transformation only scaling
test_set = test_datagen.flow_from_directory(
        'C:\\Users\\Chandramouli\\.spyder-py3\\CNN\\dataset\\test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
#Building the CNN
#Initializing the CNN
cnn=tf.keras.models.Sequential()#intialize cnn as sequential layers
#step 1-convo layer
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))
#filters are feature detectors,kernal_sizen is the size of the filter(fea detec) which we choose as 3x3,input shape 64x64 pixels since colored rgb value 3 is given for bw 1 should be given
#step 2-pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
#pool_size=pool detector which compares with convol layer and select max value in those pixels to reduce features and to preserve spatially important features
#strides-shifting no to move pool detector from one frame to another frame
#adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#step 3 -flattening
cnn.add(tf.keras.layers.Flatten())
#step 4-full connection adding ann
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
#step 5-op layer
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
#for mulitclass we need to use softmax activation fn here binary so we are using sigmoid


#training the CNN

#compiling the CNN
#connecting to a optimizer loss fn,error metric
cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#training the cnn on the training set and evaluating it on the test set below
cnn.fit(x=train_set,validation_data=test_set,epochs=25)

#making a single prediction
import numpy as np
from keras.preprocessing import image
test_image=image.load_img('C:\\Users\\Chandramouli\\.spyder-py3\\CNN\\dataset\\single_prediction\\cat_or_dog_4.jpg',target_size=(64,64))#image in predic should be same size as the ip
#predict need ip as array so we need to convert this test image
test_image=image.img_to_array(test_image)
#addin extra dimiension in prediction to match batchsize given before axis=0(batch as 1st dimension)
test_image=np.expand_dims(test_image,axis=0)
result=cnn.predict(test_image)
#to find if 0 is cat or dog
train_set.class_indices
if result[0][0]==1:#0 is batch index 0 is the only image in test
    prediction='dog'
else:
    prediction='cat'

print(prediction)


pred=cnn.predict(test_set)
test_set.class_indices

pred1=cnn.predict_classes(test_set)
import pandas as pd
pred1=pd.DataFrame(pred1)

pred1[0]=pred1[0].map({True:'dog',False:'cat'})


    

    
    

    

    

