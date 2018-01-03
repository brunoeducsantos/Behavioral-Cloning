import cv2
import csv
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D,MaxPooling2D
from keras.layers import Cropping2D
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
import pandas as pd

#read images from file
def read_file(path):
    df= pd.read_csv(path)
    return df
#split data into train and validation samples
def get_split_data(path):
    from sklearn.model_selection import train_test_split
    df = read_file(path)
    df.columns = ['center','left','right','steering','throttle','break','spead']
    train_samples, validation_samples = train_test_split(df, test_size=0.2)
    return train_samples,validation_samples

#Pre-processing data

#Normalization of pictures 
def normalization(img):
    img= img / 255. - 0.5
    return img
#general pre-processing data
def preprocessing(img):
     #convert to gray scale
    img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #resize image to 64X64
    img = cv2.resize(img,(64, 64))
    #cropp image 
    img= img[25:50,]
    #resize image to 32x32
    img = cv2.resize(img,(32, 32))
    #normalization of data
    img = normalization(img)
    img = img.reshape(32,32,1)
    return img

#Data augmentation

import sklearn
#augment brigthness
def augment_brightness_camera_images(image):
    random_bright = .25+np.random.uniform()
    return np.where((255 - image) < random_bright*image,255,image*random_bright)

#flip image
def flip(image,steering):
# decide whether to horizontally flip the image:
    flip_prob = np.random.random()
    if flip_prob > 0.5:
    # flip the image and reverse the steering angle
        steering = -1*steering
        image = cv2.flip(image, 1)
    return image,steering
#generation of new image
def get_augment_row(row):
    steering = row['steering']
    
    camera = np.random.choice(['center', 'left', 'right'])
    
    if camera == "left":
          steering += 0.05 # or any other constant value which works well
    elif camera == "right":
         steering -= 0.05
        
    if "\\" in row[camera]:
        filename= row[camera].split('\\')[-1]
    else:
        filename= row[camera].split('/')[-1]
    
    current_path = 'data/IMG/'+filename
    image = cv2.imread(current_path)
    image= preprocessing(image)
    
    image,steering= flip(image,steering)
    image = augment_brightness_camera_images(image)
    
    return image,steering
#process of data augmentation
def aug_data(samples,n_times):
    images = []
    angles = []
    
    for j in range(n_times):
        for index, row in samples.iterrows():
            img, steer = get_augment_row(row)
            img = img.reshape(32,32,1)
            images.append(img)
            angles.append(steer)
    X_train = np.array(images)
    y_train = np.array(angles)
    return X_train,y_train

#Generator
def get_data_generator(X_train,y_train, batch_size=32):
    num_samples = len(X_train)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            start = offset
            end = offset+batch_size
            X_batch,y_batch= X_train[start:end],y_train[start:end]
            images = []
            angles = []
            for index, row in zip(X_batch,y_batch):
                images.append(index)
                angles.append(row)

            X_trains = np.array(images)
            y_trains = np.array(angles)
            yield sklearn.utils.shuffle(X_trains, y_trains)


def run_model(X_train,y_train,X_test,y_test):
    model = Sequential()
    #Convolution 3X3 , depth =32
    model.add(Convolution2D(32,3,3, subsample=(1,1), activation='relu',input_shape=(64,64,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
     #Convolution 3X3 , depth =48
    model.add(Convolution2D(48,3,3, subsample=(1,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
     #Convolution 3X3 , depth =64
    model.add(Convolution2D(64,3,3, subsample=(1,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    
    model.add(Flatten())
    
    model.add(Dense(1164,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(10,activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1))
    
    model.compile(loss='mse',optimizer= 'adam')
    
    model.fit_generator( get_data_generator(X_train,y_train, batch_size=128), samples_per_epoch= 40000, validation_data=get_data_generator(X_test,y_test, batch_size=128),nb_val_samples=8000, nb_epoch=3)
    model.save('model.h5')
    
def main():
    #Dataset summary and exploration
    
    train_samples, validation_samples= get_split_data('data/driving_log.csv')
    train_samples = train_samples[(train_samples['steering']>0) | (train_samples['steering']<0)]
    train_samples = train_samples[(train_samples['steering'] <= 0.85) & (train_samples['steering']>=-0.85)]
    validation_samples = validation_samples[(validation_samples['steering']>0) | (validation_samples['steering']<0)]
    validation_samples = validation_samples[(validation_samples['steering'] <= 0.85) & (validation_samples['steering']>=-0.85)]


    X_train,y_train= train_samples[['center','left','right']],train_samples['steering']
    X_test,y_test= validation_samples[['center','left','right']],validation_samples['steering']

 
    # TODO: Number of training examples
    n_train = len(X_train)

    # TODO: Number of testing examples.
    n_test = len(X_test)

    image_shape = X_train.shape[0]

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    
    #Data augmentation
    X_train,y_train= aug_data(train_samples,20)
    X_test,y_test= aug_data(validation_samples,20)
    
    #run the model
    run_model(X_train,y_train,X_test,y_test)
if __name__ == "__main__":
    main()


