import tensorflow as tf
import tensorflow_datasets as tfds

# Load MNIST dataset
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

#convert pixel values to the range [0,1]
mnist_train=ds_train.map(lambda x, y: (tf.cast(x,tf.float32)/255.0, y))
mnist_test=ds_test.map(lambda x,y:(tf.cast(x,tf.float32)/255.0, y))

# batch and shuffle the dataset
batch_size=32
mnist_train=mnist_train.shuffle(buffer_size=len(mnist_train)).batch(batch_size)
mnist_test=mnist_test.batch(batch_size)
print(len(mnist_test))

# Define your model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(mnist_train, epochs=5, validation_data=mnist_test)