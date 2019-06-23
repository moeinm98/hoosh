
from utils import mnist_reader
import numpy as np


def perceptron_belong(img):
    global w
    MAX = -100000000000
    I = 0
    for i in range(len(w)):
        if MAX < np.dot(img, w[i][1:]) + w[i][0]:
            MAX = np.dot(img, w[i][1:]) + w[i][0]
            I = i
    return I


def perceptron_check(img, label):
    global w
    guessed_label = perceptron_belong(img)
    if guessed_label != label:
        w[label][1:] = w[label][1:] + img
        w[guessed_label][1:] = w[guessed_label][1:] - img


images, labels = mnist_reader.load_mnist('data', kind='train')
# X_test, y_test = mnist_reader.load_mnist('data', kind='t10k')

print(((images[1])))
from PIL import Image

w = [np.zeros(785) for i in range(10)]



for i in range(len(images)):
    print(i)
    perceptron_check(np.reshape(images[i], (784,)), labels[i])


# w, h = 512, 512
# data = np.zeros((h, w, 3), dtype=np.uint8)
# data[256, 256] = [255, 0, 0]
# img = Image.fromarray(X_train[0])
# img.show()
#
#
# from matplotlib import pyplot as plt
# plt.imshow()
# plt.show()