import numpy as np
# import matplotlib.pyplot as plt
import sys
import cv2
import random
import os
import pickle
import time
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

# Setup do model
DATADIR = "/Users/gustavoleonato/ExternalHD/CellData/chest_xray/train/"
DATADIR2 = "/Users/gustavoleonato/ExternalHD/CellData/chest_xray/test/"
CATEGORIES = ["NORMAL","PNEUMONIA"]
IMG_SIZE = 150
EPOCHS = 20
BATCH_SIZE = 32

# Load TRAINING data images to array
training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # caminho para o dir de cats e dogs
        class_num = CATEGORIES.index(category)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
                # print(class_num)
                # print(path, category)
            except Exception as e:
                # print("Exception!")
                pass
        
create_training_data()

# Load TEST data images to array
test_data = []
def create_test_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR2, category) # caminho para o dir de cats e dogs
        class_num = CATEGORIES.index(category)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                test_data.append([new_array,class_num])
                # print(class_num)
                # print(path, category)
            except Exception as e:
                # print("Exception!")
                pass

create_test_data()

# Print array lenght
print(len(training_data))
print(len(test_data))

# Shuffle Training data
random.shuffle(training_data)
random.shuffle(test_data)

# Training DATA to numpy array
X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)
   
X = np.array(X).reshape(-1, IMG_SIZE,IMG_SIZE, 1)

# TEST DATA to numpy array
T = []
t = []

for features2, label2 in test_data:
    T.append(features2)
    t.append(label2)
   
T = np.array(T).reshape(-1, IMG_SIZE,IMG_SIZE, 1)

# load the data  
(x_train, y_train) = (X,y)
(x_test, y_test) = (T,t)
print(x_train,y_train)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Number of classes
num_classes = 2

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# MODELING testes

dense_layers = [1]
conv_layers = [5]
layer_size = [32]

for dense_layer in dense_layers:
    for layer_size in layer_size:
        for conv_layer in conv_layers:
            ModelName = "{}-conv-{}-nodes-{}-dense-drop-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(ModelName)

            # TensorBoard
            tensorboard = TensorBoard(log_dir='logs/{}'.format(ModelName))

            model = Sequential()
            model.add(Conv2D(layer_size, (3, 3),
                            input_shape=x_train.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.25))

            model.add(Flatten())

            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))
                
            model.add(Dense(num_classes))
            model.add(Activation('softmax'))

            # Model optimizer/loss

            model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

            # Training
            model.fit(x_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(x_test, y_test),
                        # shuffle=True,
                        callbacks=[tensorboard])


# save_dir = os.path.join(os.getcwd(), 'saved_models')
# model_name = 'T3CNN4RMS.h5'

# model_path = os.path.join(save_dir, model_name)
# model.save(model_path)
# print('Saved trained model at %s ' % model_path)


# # Score trained model.
# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])