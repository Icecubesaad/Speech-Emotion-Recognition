# Speech Emotion Recognition (SER) Model

## Overview

Welcome to my Speech Emotion Recognition (SER) project! I built this model to explore the real-world applications of **Speech Emotion Recognition**. This project leverages the power of **Convolutional Neural Networks (CNNs)** to classify speech into different emotions.

While AI can’t entirely replace human emotional understanding, it can give us a glimpse into the emotional state of a user by analyzing their speech. This is the primary goal of my Speech Emotion Recognition Model.

## Problem Statement

Emotions are an essential part of human interaction, and in the age of AI, we are increasingly interested in machines' ability to understand these emotions. Speech Emotion Recognition (SER) is the task of classifying emotions from audio signals (speech). This model attempts to identify emotions such as **happy, sad, angry, and neutral** from human speech.

## Dataset

The dataset used for training and testing this model was sourced from **Dmytro Babko on Kaggle**. It contains **12.2k WAV files** with labeled emotions. The dataset includes emotions like **happy**, **sad**, **angry**, **fearful**, and **neutral**.

- **Dataset Source**: [Speech Emotion Recognition Dataset (Kaggle)](https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en)

## Key Features Extracted from Audio

To make sense of the audio data, I extracted several important features using the **Librosa** library. These features help the model distinguish between different emotions in speech:

- **Zero-Crossing Rate (ZCR)**: Indicates the rate at which the signal changes sign. It helps to identify changes in the signal.
- **Chroma STFT**: Captures pitch and harmonic information, useful for detecting emotional patterns.
- **MFCC (Mel-Frequency Cepstral Coefficients)**: Represents the timbral characteristics of speech and is crucial for emotion recognition.
- **Mel Spectrogram**: Represents sound frequencies over time, highlighting the spectral content of the audio.
- **Root Mean Square (RMS)**: Tracks the intensity of the audio signal.

## Model Architecture

For the model architecture, I built a **Convolutional Neural Network (CNN)** that processes the extracted audio features. Here's why CNNs are perfect for SER:

- **Conv1D Layers**: These layers detect local patterns in sequential audio data, which helps recognize speech patterns.
- **MaxPooling Layers**: These reduce the dimensionality and computational cost while retaining key features.
- **Dropout Layers**: To prevent overfitting during training.
- **Dense Layers**: Fully connected layers that output the final prediction, classifying the speech into one of the predefined emotion categories.

The CNN was designed as follows:

1. **Conv1D + MaxPooling1D**: The first layers are responsible for capturing features from the sequential audio data.
2. **Dropout**: Regularization technique to prevent overfitting.
3. **Flatten**: Converts the 3D matrix output from the convolutional layers into a 1D vector for the dense layers.
4. **Dense Layer**: Outputs the final predicted emotion.

### CNN Architecture Summary:

- **Input Shape**: `(X_train.shape[1], 1)`
- **Conv1D Layers**: 256 filters, kernel size of 5
- **MaxPooling1D Layers**: Pool size of 5
- **Dropout Layers**: 0.2 and 0.3 dropout to prevent overfitting
- **Dense Layers**: 32 neurons with ReLU activation and 8 neurons for emotion classification (Softmax activation)

## Results

The model was trained on the dataset and achieved an accuracy of **71%**. I’m continuously working on improving the model performance.

## How to Use the Model

You can easily run the code to train and test the model using the provided dataset. Here's a quick guide on how to use the model:

### 1. Clone the Repository

```bash
git clone https://github.com/Icecubesaad/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition
