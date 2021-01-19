#needed imports
import numpy as np
import os
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


def Load_data(directory):
    #images list and labels list to be able to append on them
    myImages = []
    ImgLabels = []
    #iterate on folders inside trainingSet
    for folder in tqdm(os.listdir(directory)):
        #join paths to go into each folder
      folder2 = os.path.join(directory, folder)
      #iterate on each image
      for files in tqdm(os.listdir(folder2)):
          #load image
          img = cv2.imread(os.path.join(folder2,files))
          #check if it's loaded successfully then add it to images list
          if img is not None:
               myImages.append(img)
               #label is the same as folder name, so i put it as folder name
               ImgLabels.append(folder)
  
    return np.array(myImages), np.array(ImgLabels)


def Train(images_array, label_array):
    #input images shape, 3 is because of RGB(three channels)
    new_shape = (28, 28, 3)
    #normalizing images
    images_array = images_array.astype('float32')
    images_array /= 255
    images_array = np.array(images_array)

    #split data to 80% for training data and 20% for testing data, shiffle is true to shuffle data before splitting
    images_array_train, images_array_test, label_array_train, label_array_test = train_test_split(images_array, label_array, test_size = 0.2, train_size = 0.8, shuffle=True)
    #change arrays from integers to binary matrix using to categorical 
    label_array_train = to_categorical(label_array_train)
    label_array_test = to_categorical(label_array_test)
    #building CNN network
    #CNN is sequential model as the output of each step is the input to the next step
    CNN_model = Sequential()
    #No. of channels:3
    #Input shape:28x28x3
    #Output shape:28x28x28
    #Conv2D is used for images, 28 filters to increase the accuracy, and 5x5 because it give better validation rate as it get the features better, same padding to avoid dimensions from vanishing
    CNN_model.add(Conv2D(28, (5, 5),padding="same", input_shape= new_shape))
    
    #No. of channels:28
    #Input shape:28x28x28
    #Output shape:14x14x28
    #maximum pooling with 2x2 (as our image is small) and stride 2 to make sure that the filter found the needed feature here to downsample the image
    CNN_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #then flatten the images to become one dimension to be able to apply functions
    
    #Input shape:14x14x28
    #Output shape:5488x1
    CNN_model.add(Flatten())
    
    #Input shape:5488x1
    #Output shape:28x1
    #(hidden layer) using same activation function as CNN lecture(Relu), 0 if the input is negative and x instead, to avoid linearity
    CNN_model.add(Dense(28, activation=tf.nn.relu))
    #function used to prevent overfitting by dropping some data
    CNN_model.add(Dropout(0.2))
    
    #Input shape:28x1
    #Output shape:10x1
    #(output layer) taking the probability and convert it to either 0 or 1, values between 0 and <0.5 become 0 and values between 0.5 and 1 become 1, the output is 10 as each output corresponds to one class
    CNN_model.add(Dense(10,activation=tf.nn.softmax))
    #we use adam optimizer as it's the most updated mode, this loss function is used when more than 2 classes are being used
    CNN_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #give the model training sets and use 10 epochs to increase accuracy, take part of the data for validation also
    CNN_model.fit(x=images_array_train, y=label_array_train, batch_size = 256, epochs=10, validation_split=0.1)
    #evaluate using testing data to get percentage of success
    CNN_model.evaluate(images_array_test, label_array_test)
    #return the array to np array to be able to use argmax
    label_array_test = np.array(label_array_test)
    #predict and print to 5 random images from training set
    for i in range(5):
        print("In image (", i+1 , "):")
        #show image
        cv2.imshow(str(np.argmax(label_array_test[i])), images_array_test[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #prediction label
        pred = CNN_model.predict_classes(images_array_test[i].reshape(1, 28, 28, 3))
        print("Predicted label: ", int(pred))
        #index of the class that gives 1
        print("True label: ", np.argmax(label_array_test[i]))
        
    #saving CNN model    
    CNN_model.save("CNN_model.h5")
    #print model summary
    print(CNN_model.summary())


#folder path
directory = "trainingSet"
#calling functions
img_array, lbl_array = Load_data(directory)
Train(img_array, lbl_array)