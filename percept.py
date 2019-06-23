from utils import mnist_reader
import numpy as np


class Perceptron:

    def __init__(self, save_train_path, epoch, data_path,C,mira=False):
        w = [np.zeros(785) for i in range(10)]
        w[0][0] = 1
        self.w = w
        self.save_train_path = save_train_path
        self.data_path = data_path
        self.epoch = epoch
        self.mira = mira
        self.C=C

    def perceptron_belong(self, img):
        MAX = -100000000000
        I = 0
        for i in range(len(self.w)):
            if MAX < np.dot(img, self.w[i][1:]) + self.w[i][0]:
                MAX = np.dot(img, self.w[i][1:]) + self.w[i][0]
                I = i
        return I

    def find_taw(self, wy, wy_star, img):
        taw=(np.dot((wy-wy_star)[1:],img)+wy[0]-wy_star[0]+1)/(2*np.dot(img,img)+2)
        return min(taw,self.C)
    def perceptron_check(self, img, label):
        guessed_label = self.perceptron_belong(img)
        taw = 1
        if guessed_label != label:
            if self.mira:
                taw = self.find_taw(self.w[guessed_label],self.w[guessed_label],img)
            self.w[label][1:] = self.w[label][1:] + taw * img
            self.w[label][0] += taw
            self.w[guessed_label][1:] = self.w[guessed_label][1:] - taw * img
            self.w[guessed_label][0] -= taw

    def train(self):
        images, labels = mnist_reader.load_mnist(self.data_path, kind='train')
        for e in range(self.epoch):
            for i in range(len(images)):
                print(i)
                self.perceptron_check(np.reshape(images[i], (784,)), labels[i])
            print(self.w)

    def test(self, test_data_path):
        images, labels = mnist_reader.load_mnist(test_data_path, kind='t10k')
        error = 0
        for i in range(len(images)):
            guessed_label = self.perceptron_belong(np.reshape(images[i], (784,)))
            if guessed_label != labels[i]: error += 1
        return 100 - error / len(images) * 100


# print(((images[1])))
from PIL import Image

perc = Perceptron('saved_path', 2, 'data',1,mira=True)
perc.train()
ratio = perc.test('data')
print(ratio)
# print(w)
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
