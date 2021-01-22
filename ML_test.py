import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

# Loading the data
data = "F:\\Nextcloud\\assignments\\Odense Tekniske Gymnasium assignments\\OTG 3.C\\Programmering\\git\\Machine_Learning_test\\ML-Test-\\images\\dataset\\training_set"
categories = ["dogs", "cats"]

for category in categories:
    path = os.path.join(data, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        #plt.imshow(img_array, cmap="gray")
        #plt.show()
        break
    break


img_size = 100
new_array = cv2.resize(img_array, (img_size, img_size))
plt.imshow(new_array, cmap = "gray")
plt.show()

training_data = []

def train_data():
    for category in categories:
        path = os.path.join(data, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

#create_training_data()

random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])

X = []
Y = []

for features, label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 1)