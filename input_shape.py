import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
import numpy as np

#define the input shape
input_shape=(32,32,3)

#define the input layer with correct input shape
input_layer=Input(shape=input_shape)

#validate input data shape before passing it to the model
def validate_input_data(data):
    if data.shape[1:]!=input_shape:
        raise ValueError(f"Input data shape {data.shape[1:]} does not match the expected shape {input_shape}")

# example uses
input_data=np.random.randn(10, *input_shape)
validate_input_data(input_data)