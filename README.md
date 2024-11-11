<img width="785" alt="Screenshot 2024-11-12 at 00 37 43" src="https://github.com/user-attachments/assets/cf3f65eb-edfc-4553-9f53-fc9ca4d242a3">



## Dog Face Emotion Recognition

This project focuses on detecting and classifying emotions from dog faces using deep learning. The core of the solution is built around a Convolutional Neural Network (CNN) that has been fine-tuned with transfer learning using the InceptionV3 model to enhance recognition accuracy.

## Project Overview

Understanding emotions in animals, particularly dogs, is an area of growing interest, with potential applications in behavioral studies, pet care, and veterinary science. This project aims to contribute to this field by developing a machine learning model that recognizes and classifies emotions from images of dog faces.

The project is structured to leverage the power of Convolutional Neural Networks (CNNs) to extract meaningful features from dog face images. We also integrated InceptionV3, a pre-trained deep learning model, using transfer learning to optimize performance and accuracy in emotion recognition.

## Dataset

The dataset contains labeled images of dog faces, each associated with different emotional expressions. The dataset is split into training, validation, and test sets to build and evaluate the model.

## Tools & Technologies

Python: Core programming language used for building the model.

TensorFlow: Framework used for implementing deep learning models.

Pandas: Used for data manipulation and analysis.

NumPy: Used for numerical operations on data.

Matplotlib: Library used for visualizing model performance (e.g., accuracy, loss).

Seaborn: Used for advanced data visualization and exploratory data analysis.

CNN: Sequential 2D Convolutional Neural Network used for emotion detection.

InceptionV3: Pre-trained model integrated for transfer learning to improve accuracy.

## Model Architecture

### 1. CNN Model
The model starts with a Sequential 2D CNN architecture.
Multiple convolutional and pooling layers are stacked to extract features from the input images.
Fully connected layers are used to interpret the features and classify the emotions.
### 2. InceptionV3 Model
A pre-trained InceptionV3 model is used to leverage transfer learning.
The model is fine-tuned on the dataset to enhance the accuracy of emotion recognition.
Transfer learning significantly reduced training time while improving performance.
