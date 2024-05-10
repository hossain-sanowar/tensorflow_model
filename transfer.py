import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.applications import VGG16

#load the pretrained model
base_model=VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

#choose the intermediate layers for feature extraction
intermediate_layer_names=['block2_pool', 'block5_pool']
intermediate_layer_outputs=[base_model.get_layer(name).output for name in intermediate_layer_names]
print(intermediate_layer_outputs)

#create a feature extractor model
feature_extractor_model =Model(inputs=base_model.input,
                               outputs=intermediate_layer_outputs)

#pass data through the feature extractor model to extract features
features =feature_extractor_model.predict(data)

#use extracted features for our task (e.g. classification, clustering)



