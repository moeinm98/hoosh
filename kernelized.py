from utils import mnist_reader
import numpy as np


class Kernelized:
    def __init__(self, save_train_path, epoch, data_path):
        w = [np.zeros(60000) for i in range(10)]
        self.w = w
        self.save_train_path = save_train_path
        self.data_path = data_path
        self.epoch = epoch

    def perceptron_belong(self, img,s=60000):
        MAX = -100000000000
        I = 0
        for i in range(len(self.w)):
            sum = 0
            for j in range(s):
                sum += self.w[i][j] * self.K(self.images[j], img)
            if MAX < sum:
                MAX = sum
                I = i
        return I

    def K(self, fxi, fx):
        return (np.dot(fxi, fx) + 2) ** 2

    def perceptron_check(self, img,i, label):
        guessed_label = self.perceptron_belong(img,s=i)
        if guessed_label != label:
            self.w[label][i] += 1
            self.w[guessed_label][i] -= 1

    def train(self):
        self.images, labels = mnist_reader.load_mnist(self.data_path, kind='train')
        for e in range(self.epoch):
            for i in range(len(self.images)):
                print(i)
                self.perceptron_check(np.reshape(self.images[i], (784,)),i, labels[i])
            print(self.w)
        np.savetxt('learned.txt', self.w, fmt='%d')

    def test(self, test_data_path):
        self.w = np.loadtxt('learned.txt', dtype=int)
        images, labels = mnist_reader.load_mnist(test_data_path, kind='t10k')
        error = 0
        for i in range(len(images)):
            guessed_label = self.perceptron_belong(np.reshape(images[i], (784,)))
            if guessed_label != labels[i]: error += 1
        return 100 - error / len(images) * 100


# print(((images[1])))
from PIL import Image
perc = Kernelized('saved_path', 1, 'data')
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
