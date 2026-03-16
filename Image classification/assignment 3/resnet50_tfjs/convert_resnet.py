import tensorflow as tf
from tensorflowjs.converters import save_keras_model

print("tensorflow", tf.__version__)
model = tf.keras.applications.ResNet50(weights="imagenet")
model.save("resnet50_keras.h5")
print("saved resnet50_keras.h5")
save_keras_model(model, "resnet50_tfjs")
print("saved resnet50_tfjs/model.json and shard")