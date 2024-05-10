import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
import numpy as np

input_shape=(32,32,3)
input_layer=Input(shape=input_shape)

def validate_test_data(test_data):
    if test_data.shape[1:] !=input_shape:
        raise ValueError(f"Input data shape {input_shape.shape} "
                         f"does not match with expected test data shape {test_data.shape}")

input_data=np.random.randn(100, *input_shape)
validate_test_data(input_data)