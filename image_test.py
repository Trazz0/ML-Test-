import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
# X = pickle.load(open("X.pickle","rb"))
# y = pickle.load(open("y.pickle","rb"))
# a = random.randint(0, 7990)
# print(X.shape)
# for i in range(10):
#     if y[a] == 0:
#         print("It is a dog")
#     else:
#         print("It is a cat")
#     plt.imshow(X[a], cmap="gray")
#     plt.show()
#     a += 1
#     # print(X[a])


# Predictions for test images

test_data = "F:\\Nextcloud\\assignments\\Odense Tekniske Gymnasium assignments\\OTG 3.C\\Programmering\\git\\Machine_Learning_test\\ML-Test-\\images\\dataset\\test_set\\combined"
categories = ["dog", "cat"]
data = []
img_size = 100
for img in os.listdir(test_data):
    img_array = cv2.imread(os.path.join(test_data, img), cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (img_size, img_size))
    data.append(new_array)
model = tf.keras.models.load_model("dog-cat-ML.model")
random.shuffle(data)
for i in range(10):
    a = random.randint(0, 1990)
    test = data[a].reshape(-1, img_size, img_size, 1)
    predictions = model.predict(test)
    print(categories[int(predictions[0][0])])
    plt.imshow(data[a], cmap="gray")
    plt.show()
