
from utils import mnist_reader
import numpy as np
images, labels = mnist_reader.load_mnist('data', kind='train')
# X_test, y_test = mnist_reader.load_mnist('data', kind='t10k')

print(((images[1])))
from PIL import Image


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