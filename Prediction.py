import keras
from keras.models import load_model
import cv2
import os
import numpy as np 

img_array = cv2.imread('VIRUS-9671740-0001.jpeg', cv2.IMREAD_GRAYSCALE)
new_array = cv2.resize(img_array,(150,150))

# Reshape
X = []
X = np.array(new_array).reshape(-1, 150, 150, 1)

print(X.shape)

model = load_model('saved_models/T3CNN.h5')

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

prediction = model.predict_classes(X)
print('0: Normal')
print('1: Pneumonia')
print(prediction)
