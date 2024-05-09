import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model=Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


model.fit(x_train, y_train, epochs=5)

test_loss, test_accuracy=model.evaluate(x_test,y_test)
print(f"test_loss:{test_loss}")
print(f"test_accuracy:{test_accuracy}")
#model.summary()
model.save("sequential_model.h5")
load_model=tf.keras.models.load_model("sequential_model.h5")
y_pred=model.predict(x_test)
print(y_pred)