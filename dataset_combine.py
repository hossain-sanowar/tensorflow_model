import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
batch_size=32

#mnist dataset
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist)=mnist.load_data()
x_train_mnist, x_test_mnist=x_train_mnist/255.0, x_test_mnist/255.0
print(x_train_mnist.shape)

#create tensorflow dataset for mnist
mnist_train_dataset=tf.data.Dataset.from_tensor_slices((x_train_mnist, y_train_mnist))
mnist_test_dataset=tf.data.Dataset.from_tensor_slices((x_test_mnist,y_test_mnist))


#cifar10 dataset
(x_train_cifar10, y_train_cifar10), (x_test_cifar10, y_test_cifar10)=cifar10.load_data()
x_train_cifar10, x_test_cifar10=x_train_cifar10/255.0, x_test_cifar10/255.0

#create tensorflow dataset for cifar10
cifar10_train_dataset=tf.data.Dataset.from_tensor_slices((x_train_cifar10, y_train_cifar10))
cifar10_test_dataset=tf.data.Dataset.from_tensor_slices((x_test_cifar10,y_test_cifar10))

#combine two dataset
combine_train_dataset=mnist_train_dataset.concatenate(cifar10_train_dataset)
combine_test_dataset=mnist_test_dataset.concatenate(cifar10_test_dataset)
print(len(combine_train_dataset), len(combine_test_dataset))
#shuffle the dataset
combine_train_dataset=combine_train_dataset.shuffle(buffer_size=100000).batch(batch_size)
combine_test_dataset=combine_test_dataset.shuffle(buffer_size=100000).batch(batch_size)

# model.fir(combine_train_dataset, epochs=10, validation_data=combine_test_dataset)