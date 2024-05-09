"""
steps:
    1. convert lowercase and remove all delimiter
    2. convert Word Index Dictionary
    3. Create Tokenized sequences
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence

"""
It seems there's an issue with importing the text_to_sequences 
function from the tensorflow.keras.preprocessing.text module. 
This function is not available directly in the tensorflow.keras.preprocessing.text module.
"""
# Sample text data
texts = [
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur."
]
# Convert text to word sequences
for text in texts:
    word_seq=text_to_word_sequence(text)
    print("\nOrginal Text: ", text)
    print("word sequence:", word_seq)

# Tokenization
tokenizer = Tokenizer(num_words=100)  # Limit vocabulary to 100 words
tokenizer.fit_on_texts(texts)

# Print output
print("Word Index Dictionary:")
print(tokenizer.word_index)

# Convert text to word sequences
sequences = tokenizer.texts_to_sequences(texts)
print("Tokenized sequences: ")
for seq in sequences:
    print(seq)


