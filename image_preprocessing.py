import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from IPython.display import display

#ImageDataGenerator for data augmentation
datagen=ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load an image from file
img=load_img('57.jpeg')

from IPython.display import display

# Display the original image
print("Original Image:")
display(img)

import matplotlib.pyplot as plt
# Display the original image using Matplotlib
plt.imshow(img)
#plt.show()

#convert image to a numpy array
img_array=img_to_array(img)
#print(f"convert image to numpy array: {img_array}")

#Reshape the image array to (1, height, width, channels) for compatibility with ImageDataGenerator

img_array=img_array.reshape((1,)+img_array.shape)
print(f"convert image to numpy array: {img_array.shape}")

# Generated augmented images
i=0
for batch in datagen.flow(img_array, batch_size=1):
    i+=1
    augmented_img=array_to_img(batch[0])
    print(f"Augmented Image {i}: ")
    display(augmented_img)
    plt.imshow(augmented_img)
    plt.show()
    if i==10: #generated 5 augmented image
        break