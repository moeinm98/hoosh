from utils import mnist_reader
import numpy as np


class Kernelized:
    def __init__(self, save_train_path, epoch, data_path):
        self.w = [np.array([]) for i in range(10)]
        self.w_index = [np.array([]) for i in range(10)]
        self.save_train_path = save_train_path
        self.data_path = data_path
        self.epoch = epoch

    def perceptron_belong(self, img):
        MAX = -100000000000
        I = 0
        for i in range(len(self.w)):
            sum = 0
            for j in range(len(self.w[i])):
                sum += self.w[i][j] * self.K(self.images[int(self.w_index[i][j])], img)
            if MAX < sum:
                MAX = sum
                I = i
        return I

    def K(self, fxi, fx):
        return (np.dot(fxi, fx) + 2) ** 2

    def perceptron_check(self, img,i, label):
        guessed_label = self.perceptron_belong(img)
        if guessed_label != label:
            self.w[label]=np.append(self.w[label],[1])
            self.w[guessed_label]=np.append(self.w[guessed_label],[-1])
            self.w_index[label]=np.append(self.w_index[label],[i])
            self.w_index[guessed_label]=np.append(self.w_index[guessed_label],[i])

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
# ratio = perc.test('data')
print(perc.w)


# print(w)
# w, h = 512, 512
# data = np.zeros((h, w, 3), dtype=np.uint8)
# data[256, 256] = [255, 0, 0]
# img = Image.fromarray(images[0])
# img.show()
#

# from matplotlib import pyplot as plt
#
# plt.imshow()
# plt.show()
