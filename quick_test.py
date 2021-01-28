from ML_test import X, y, img_array
import pickle
import matplotlib.pyplot as plt
pickle_in = open("X.pickle", "rb")
X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))
a = 0
#print(type(img_array))
plt.imshow(img_array, cmap="gray")
plt.show()
if y[a] == 0:
    print("It is a dog")
else:
    print("It is a cat")
print(X[a],y[a])