"""
1. convert pad sequence to the same length
2. here maximum length 5
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

#sample sequences
sequences=[
    [1,2,3],
    [4,5],
    [6,7,8,9],
    [10,11,12,13,14]
]
# Pad sequences to the same length
padded_sequences=pad_sequences(sequences, maxlen=5, padding='post', truncating='post', value=0)

print("Original Sequences:")
for seq in sequences:
    print(seq)

print("\nPadded Sequences:")
for padded_seq in padded_sequences:
    print(padded_seq)