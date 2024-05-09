import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import GlorotNormal,HeNormal, RandomNormal

# Dummy data for demonstration
import numpy as np
x_train = np.random.random((1000, 784))
y_train = np.random.randint(10, size=(1000,))

#def model
def model_init(initializer):
    model=Sequential([
        Dense(64, activation='relu', kernel_initializer=initializer, input_shape=(784,)),
        Dense(10, activation='softmax')
    ])

    return model

model_glorot=model_init(initializer=GlorotNormal())
model_he=model_init(initializer=HeNormal())
model_random=model_init(initializer=RandomNormal(mean=0.0, stddev=0.05))

# initilize =[GlorotNormal(), HeNormal(), RandomNormal(mean=0.0, stddev=0.05)]
for model in [model_glorot,model_he, model_random]:
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)



