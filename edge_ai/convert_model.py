import tensorflow as tf
from tensorflow import keras
import os

# Disable oneDNN logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load the model without compile since we're only converting it
model = keras.models.load_model("model/recyclable_classifier.keras", compile=False)

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('tflite/model.tflite', 'wb') as f:
    f.write(tflite_model)


print("TFLite model conversion complete.")
