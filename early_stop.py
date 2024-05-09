import os
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np

#genrated dummy data
x_train=np.random.random((1000,784))
y_train=np.random.randint(2, size=(1000,10))
x_val=np.random.random((200,784))
y_val=np.random.randint(2, size=(200,10))
x_test=np.random.random((200,784))
y_test=np.random.randint(2,size=(200,10))


#create a seguential model
model = Sequential([
    Dense(units=64, activation='relu', input_shape=(784,)),
    Dense(units=10, activation='softmax')
])

#model compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Display the model summary
model.summary()

#train the model
# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint(filepath='best_model.keras', monitor='val_acc', save_best_only=True)
#reduce_lr=ReduceLROnPlateau()
# Train the model with callbacks
model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), callbacks=[early_stopping,
                                                                                  model_checkpoint])
                                                                                  #reduce_lr])

#evaluate the model
loss, accuracy=model.evaluate(x_test, y_test)
print(f"loss: {loss}, accuracy: {accuracy}")

#save the model
model.save('early_model.keras')

#load the model
loaded_model=load_model('early_model.keras')