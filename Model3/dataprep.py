import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

(tmp_x, tmp_y),(payload_x, payload_y) = mnist.load_data()
output_dim = len(np.unique(payload_y))
payload_x_reshape = payload_x.reshape(10000, 784).astype('float32')
payload_x_normalize = payload_x_reshape / 255

payload_train_x, payload_test_x, payload_train_y, payload_test_y = train_test_split(payload_x_normalize, payload_y, test_size=0.33, random_state=22)

payload_train_y_oneHot = np_utils.to_categorical(payload_train_y)
payload_test_y_oneHot = np_utils.to_categorical(payload_test_y)
