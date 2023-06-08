import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Preprocess the new data
new_image_path = "new_image.jpg"
new_image = load_img(new_image_path, grayscale=True, target_size=(48, 48))
new_image = img_to_array(new_image) / 255.0
new_image = np.expand_dims(new_image, axis=0)

# Load the trained model
model = tf.keras.models.load_model("saved_model.h5")

# Make predictions on the new data
predictions = model.predict(new_image)

# Process the predictions
emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

predicted_label = np.argmax(predictions, axis=1)[0]
predicted_emotion = emotion_labels[predicted_label]

print("Predicted Emotion:", predicted_emotion)
