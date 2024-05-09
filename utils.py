import tensorflow as tf
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from pydot import dot_parser
import pydot

#data
x_train=[[1,2,3],
         [4,5,6],
         [7,8,9]]
y_train=[0,1,0]

#convert class vector to binary class matrix
y_train_categorical=to_categorical(y_train)
print("Original class labels: ")
print(y_train)
print("\nOne-hot encoded labels:")
print(y_train_categorical)

#model
model=Sequential([
    Dense(64, activation='relu', input_shape=(3,)),
    Dense(2, activation='softmax')
])

import pydot

try:
    # plot the model architecture
    plot_model(model, to_file='model_plot.png', show_shapes=True,show_layer_names=True)
    print("\nModel architecture plotted and saved as model_plot.png")

except (OSError, Exception) as e:
    # Handle the exceptions
    print("Error occurred:", e)
