import tensorflow as tf

from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, imdb

#load dataset

def load_data(data):
    if data==imdb:
        (x_train, y_train), (x_test, y_test) = data.load_data()
        length = len(x_train), len(y_train), len(x_test), len(y_test)
        return length
    else:
        (x_train, y_train), (x_test, y_test) = data.load_data()
        shape = x_train.shape, y_train.shape, x_test.shape, y_test.shape
        length = len(x_train)
        return shape, length

data_list=[mnist, fashion_mnist, cifar10, imdb]

# for data in data_list:
#     data_=load_data(data)
#     print(data_)

data_=list(map(load_data, data_list))
print(data_)


