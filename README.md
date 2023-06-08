# Facial Expression Recognition with Convolutional Neural Networks

This project trains a Convolutional Neural Network (CNN) model for Facial Expression Recognition (FER) using the FER2013 dataset. The FER2013 dataset contains images of faces labeled with one of seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Prerequisites

- pandas
- numpy
- tensorflow
- scikit-learn

You can install the required Python packages by running the following command:

```
pip install -r requirements.txt
```

## Usage
1. Prepare the dataset: Download the FER2013 dataset from [Kaggle](https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition).
2. Train the model: Run the [emotion_detection.py]() script to train the facial expression recognition model.

```
python emotion_detection.py
```
3. Predict on new data:
    - Place the new image in the project directory.
    - Update the file path in the [predict_emotion.py]() script to point to the new image.
    - Run the [predict_emotion.py]() script to predict the emotion in the new image.

```
python predict_emotion.py
```

## Files
- The [emotion_detection.py]() script trains the model using the FER dataset and saves the trained model to a file (saved_model.h5).

- The [predict_emotion.py() script loads the trained model from the saved file and predicts the emotion in a new image.

- The [requirements.txt]() file contains a list of Python packages and their versions required for this project. You can install them using the command mentioned in the "Prerequisites" section.
