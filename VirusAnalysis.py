'''This program was Made by Dēv Mehra Age 13 in March 2020
The dataset is from https://github.com/ieee8023/covid-chestxray-dataset

'''


import numpy as np 
import pandas as pd 
import tensorflow.compat.v1 as tf
from sklearn import preprocessing, neighbors
import sklearn.model_selection
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow import keras

tf.disable_v2_behavior()

#Reading data from dataset
datadirectory=r"C:\Users\Dev Mehra\Documents\Python Projects\Final Programs\Final Programs\datasets\covid-chestxray-dataset-master\covid-chestxray-dataset-master\images"
df = pd.read_csv(r"C:\Users\Dev Mehra\Documents\Python Projects\Final Programs\Final Programs\datasets\covid-chestxray-dataset-master\covid-chestxray-dataset-master\metadata.csv")
df.replace("?",-99999,inplace=True)
df.set_index('filename', inplace=True)

#Lists for storing file data later on
labels=[]
data = []



#Accessing and storing data from dataset
for image in os.listdir(datadirectory):
   #In case the image is not found
   try:
      #Converting and scaling grayscaled images into array so they can be read
      imagearray = cv2.imread(os.path.join(datadirectory,image),cv2.IMREAD_GRAYSCALE)
      newimagearray=cv2.resize(imagearray,(50,50))


      try:
          #Finding is the value column containing all the diagnoses results corresponding with each image
          #Try and except are needed because there are a few more images than rows in the data. 
          #This line allows for the program to find the finding corresponding to the current image
          #This allows for images and their finding values to be ordered in separate lists in a way that each image has the same index/location as its corresponding finding
         label=df.loc[image]['finding']

      #If the image is not found in metadata, it will skip the image and go to the next, allowing the problem of extra images without a corresponding value to be solved
      except:
         continue
      
      else:
         #Adding the images and findings to separate lists, but the images have the same index as their corresponding finding
         data.append(newimagearray)
         labels.append(label)
         
   #Skips and resets the loop if the image is not found
   except:
      continue
   
#Converting data into an array and allowing the program to reshape the array how it needs.
data = np.array(data)
data = data.reshape(len(data),-1)

#Creating the testing and training modules using the data and labels values
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(data,labels,test_size=0.1)

#x_train and x_test contain the arrays for the images
#y_train and y_test contain the corresponding labels for the images


'''Every number is from 1 and 255.
   Dividing by 255 allows the range of numbers to decrease to between 0 and 1.
   This allows the accuracy to increase.'''
x_train = x_train/255
x_test = x_test/255

#The lists will become the new label containers
y_train_list = []
y_test_list = []

#The labels need to be converted into numbers for the model to read them
for i in y_train:
   
   if i == 'COVID-19':
      y_train_list.append(0)
   if i == 'ARDS':
      y_train_list.append(1)
   if i == 'SARS':
      y_train_list.append(2)
   if i == 'Pneumocystis':
      y_train_list.append(3)
   if i == 'Streptococcus':
      y_train_list.append(4)
   if i == 'No Finding':
      y_train_list.append(5)
   
for i in y_test:
   
   if i == 'COVID-19':
      y_test_list.append(0)
   if i == 'ARDS':
      y_test_list.append(1)
   if i == 'SARS':
      y_test_list.append(2)
   if i == 'Pneumocystis':
      y_test_list.append(3)
   if i == 'Streptococcus':
      y_test_list.append(4)
   if i == 'No Finding':
      y_test_list.append(5)


#Creating the model
model = keras.Sequential([keras.layers.Flatten(input_shape=(2500,)),
keras.layers.Dense(256,activation='relu'),
keras.layers.Dense(6,activation='softmax')])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#200 epochs
history = model.fit(x_train,y_train_list,epochs=200)

#Calculating the accuracy for test data
loss, accuracy = model.evaluate(x_test,y_test_list)
print("Accuracy:",str(accuracy*100)+"%")
print("Loss:",loss)

# #Graphing the Accuracy and Loss through the epochs
# plt.plot(history.history['acc'])
# plt.plot(history.history['loss'])
# plt.title('Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Accuracy','Loss'],loc='upper left')
# plt.show()

#Replace the image path with whatever the path is for the image you want to test
testimage = cv2.imread(r'C:\Users\Dev Mehra\Documents\Python Projects\Final Programs\Final Programs\007da8b578e9e5a8a9be8fb1f8ac22_jumbo.jpg',cv2.IMREAD_GRAYSCALE)
testimage=cv2.resize(testimage,(50,50))
testimagearray = np.array(testimage)
testimagearray = np.reshape(testimagearray,(1,2500))

predictions = model.predict(testimagearray)

#Highest probability predicted by model
prediction = np.argmax(predictions[0])
if prediction == 0:
    print("There is a "+str(accuracy*100)+"% "+" chance you have COVID-19 Pneumonia")

if prediction == 1:
    print("There is a "+str(accuracy*100)+"% "+" chance you have ARDS Pneumonia")

if prediction == 2:
    print("There is a "+str(accuracy*100)+"% "+" chance you have SARS Pneumonia")

if prediction == 3:
    print("There is a "+str(accuracy*100)+"% "+" chance you have Pneumocystis Pneumonia")

if prediction == 4:
    print("There is a "+str(accuracy*100)+"% "+" chance you have Streptococcus Pneumonia")

#No Finding does not work very well, as the dataset only has 1 or 2 cases of it
if prediction == 5:
    print("There is a "+str(accuracy*100)+"% "+" chance you have no infection")

