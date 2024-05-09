import numpy as np

arr=np.arange(100) # Array with 100 elements
print(f"original array: {arr}")

arr_reshape=arr.reshape(-1, 10)  # Reshape with -1 for the first dimension, and 10 for the second dimension
print(f"original array: {arr_reshape}")

# Reshape with -1 for the first dimension, 5 for the second dimension, and 2 for the third dimension
arr_reshape=arr.reshape(-1, 5, 2)
print(f"original array: {arr_reshape}")