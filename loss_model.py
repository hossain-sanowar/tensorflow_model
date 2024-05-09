import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy, MeanSquaredError, BinaryCrossentropy

# Define the model architecture
model_sparse_categorical = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])

model_mean_squared = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])

model_binary_crossentropy = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(1, activation='sigmoid')
])

# Compile the models with different loss functions
model_sparse_categorical.compile(optimizer=Adam(learning_rate=0.001), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
model_mean_squared.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError(), metrics=['accuracy'])
model_binary_crossentropy.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryCrossentropy(), metrics=['accuracy'])

# Dummy data for demonstration
import numpy as np
x_train = np.random.random((1000, 784))
y_train_sparse_categorical = np.random.randint(10, size=(1000,))
y_train_binary_crossentropy = np.random.randint(2, size=(1000,))

# Train the models
model_sparse_categorical.fit(x_train, y_train_sparse_categorical, epochs=10, batch_size=10, validation_split=0.2)
model_mean_squared.fit(x_train, y_train_sparse_categorical, epochs=10, batch_size=10, validation_split=0.2)
model_binary_crossentropy.fit(x_train, y_train_binary_crossentropy, epochs=10, batch_size=10, validation_split=0.2)