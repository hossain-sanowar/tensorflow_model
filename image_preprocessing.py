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
    #plt.show()
    if i==10: #generated 5 augmented image
        break

# Load original image from data folder like train, test folder
# load images from directory
train_generator=datagen.flow_from_directory(
    'train/',
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical' #class_mode='binary', the generator will return 1D numpy arrays of binary labels (0s and 1s) for binary classification problems
)
#class_mode='categorical'. This will return 2D numpy arrays of one-hot encoded labels, where each row corresponds to one image and each column corresponds to one class. The value at a particular row-column combination will be 1 if the image belongs to that class, and 0
# # Get a batch of images and labels from the generator
# images, labels = train_generator.next()
#
# # Display the first few images from the batch
# plt.figure(figsize=(10, 10))
# for i in range(9):
#     plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i])
#     plt.title(f'Label: {labels[i]}')
#     plt.axis('off')
# plt.show()

# Display the first batch of images
for i in range(len(train_generator)):
    images, labels = train_generator[i]  # Get a batch of images and labels
    plt.figure(figsize=(10, 10))
    for j in range(len(images)):
        plt.subplot(3, 3, j + 1)
        plt.imshow(images[j])
        plt.title(f'Label: {labels[j]}')
        plt.axis('off')
    plt.show()
    break  # Only display the first batch