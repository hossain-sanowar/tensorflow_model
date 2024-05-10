import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
import numpy as np

input_shape = (32,32,3)

def predict(test_data):
    return test_data*2

def validate_output_shape(output_data):
    if output_data.shape[1:] != input_shape:
        raise ValueError(f"Output data shape {output_data.shape[1:]} does not match expected input shape {input_shape}")


test_data=np.random.randn(10, *input_shape)

#perform interence with the test data
#prediction results
output_data=predict(test_data)

#validate out_put shape
#get the prediction result which shape can be verified
#we can verify the prediction shape results
validate_output_shape(output_data)



