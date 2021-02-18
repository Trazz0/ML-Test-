import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random

# Predictions for test images

test_data = "\\ML-Test-\\images\\dataset\\test_set\\combined"
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
