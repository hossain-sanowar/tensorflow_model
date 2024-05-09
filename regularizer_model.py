import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import L1, L2, L1L2

#define dataset
import numpy as np
x_train = np.random.random((1000, 784))
y_train=np.random.randint(10, size=(1000,))

#define model

def regularize_model(regularizer):
    model = Sequential([
        Dense(64, activation='relu', kernel_regularizer=regularizer, input_shape=(784,)),
        Dense(10, activation='softmax')
    ])

    return model
param_list=[L1(0.01), L2(0.01), L1L2(l1=0.01, l2=0.01)]
regularizer_list=list(map(regularize_model,param_list))

for model in regularizer_list:
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

