**Brain Tumor Detection using CNN**

This project uses Convolutional Neural Networks (CNN) to classify brain MRI images as either tumor or no tumor. The model was trained on 400+ MRI images and achieved high performance in terms of accuracy, precision, recall, and F1-score. The project leverages Keras and TensorFlow for model building and training.

**Table of Contents**

Project Overview
Dataset
Model Architecture
Results
Technologies Used
Installation
Usage
Contributing
License
Project Overview
This project focuses on detecting brain tumors from MRI images using a deep learning approach. By applying CNNs, we classify MRI images into two categories: tumor and no tumor. Various image pre-processing techniques, such as resizing and normalization, were employed to improve model performance.

**Dataset**

The dataset contains 400+ labeled MRI images, split into two classes:

Tumor: MRI images that show brain tumors.
No Tumor: MRI images without tumors.
The dataset was pre-processed to ensure consistency in image size (128x128) and format (RGB).

**Model Architecture**

The Convolutional Neural Network (CNN) consists of the following layers:

Conv2D + ReLU Activation: Extracts features from the input image.
Batch Normalization: Stabilizes and accelerates training.
MaxPooling2D: Reduces dimensionality while preserving essential features.
Dropout: Reduces overfitting by randomly dropping neurons during training.
Dense (Fully Connected): Final layers for classification with a softmax activation function.
Model Highlights:
Input: 128x128 RGB images
Optimizer: Adam
Loss Function: Categorical Crossentropy
Metrics: Accuracy, Precision, Recall, F1-Score
Results
The model achieved the following performance metrics on the test set:

**Accuracy: 93%
Precision: 91%
Recall: 92%
F1-Score: 91.8%**


**Technologies Used**

Python
Keras & TensorFlow (for deep learning)
NumPy & Pandas (for data handling)
Matplotlib (for data visualization)
Scikit-learn (for metrics and data splitting)
