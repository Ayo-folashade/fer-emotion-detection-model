import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the FER dataset
data = pd.read_csv('fer2013.csv')

# Splitting into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data['pixels'], data['emotion'], test_size=0.2, random_state=42)

# Preprocessing the data
# Convert the pixel values to numpy arrays and normalize them
X_train = np.array([np.fromstring(image, dtype=int, sep=' ') for image in X_train]) / 255.0
X_val = np.array([np.fromstring(image, dtype=int, sep=' ') for image in X_val]) / 255.0

# Reshape the image data to have a shape of (-1, 48, 48, 1) to fit the Conv2D layers
X_train = X_train.reshape(-1, 48, 48, 1)
X_val = X_val.reshape(-1, 48, 48, 1)

# Convert the emotion labels to one-hot encoded vectors using the num_classes=7
y_train = tf.keras.utils.to_categorical(y_train, num_classes=7)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=7)

# Create an ImageDataGenerator object to perform data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,  # Randomly rotate the images by 10 degrees
    width_shift_range=0.1,  # Randomly shift the width of the images by 0.1
    height_shift_range=0.1,  # Randomly shift the height of the images by 0.1
    shear_range=0.2,  # Apply shearing transformation to the images
    zoom_range=0.2,  # Randomly zoom in or out on the images
    horizontal_flip=True,  # Randomly flip the images horizontally
    fill_mode='nearest'  # Fill any missing pixels after transformations with the nearest available pixel
)

# Fit the data augmentation generator on the training data
# This step prepares the generator to perform data augmentation during training
datagen.fit(X_train)

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # Max pooling layer to reduce the spatial dimensions
model.add(Dropout(0.25))  # Regularization technique to prevent overfitting
model.add(Flatten())  # Flatten the data for the fully connected layers
model.add(Dense(128, activation='relu'))  # Fully connected layer with 128 units and ReLU activation
model.add(Dropout(0.5))  # Regularization technique
model.add(Dense(7, activation='softmax'))  # Output layer with 7 units for 7 emotion classes and softmax activation

# Define the loss function, optimizer, and metrics for model compilation
model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Train the model using the data augmentation generator and validate on the validation data
model.fit(datagen.flow(X_train, y_train, batch_size=64), epochs=50, validation_data=(X_val, y_val))

# Save the trained model to a file named 'emotion_detection_model.h5'
model.save('emotion_detection_model.h5')
