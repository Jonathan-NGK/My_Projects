# Face Recognition for Biometric Application by Jonathan NGalamulume Kalonji

## Overview

This project presents a robust facial recognition system designed for biometric authentication in both real-time security identification and web-based login applications. It combines state-of-the-art deep learning techniques, data augmentation, and transfer learning to deliver high accuracy in diverse and challenging conditions such as varying lighting, poses, and partial occlusions.

## Features

- 🧠 **Transfer Learning with FaceNet-512**: Utilizes a pre-trained FaceNet model to generate high-dimensional embeddings for face images.
- 🫲 **1D-CNN Classifier**: Custom-designed convolutional neural network for identity classification based on facial embeddings.
- 🔄 **Extensive Data Augmentation**:
  - Elliptical masking for occlusion simulation
  - Horizontal flipping
  - Affine transformations (rotation)
  - Brightness adjustment
  - Gaussian blur
- 📊 **Evaluation Metrics**: Includes accuracy, precision, recall, and F1-score for performance assessment.
- 🌐 **Web App Integration**: Designed to support both offline and web-based authentication workflows.
- 🎥 **Real-Time Face Detection**: Built-in support for live video streams using OpenCV and Haar cascades or MTCNN/RetinaFace.

## Dataset

A custom dataset was created consisting of:
- 9 unique users
- 50 original RGB images per user (450 images)
- Augmented using 5 different transformations, expanding the dataset to approximately **45,000 images**

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Jonathan-NGK/My_Projects.git
   cd Face_Recognition_for_Biometric_Application
   ```

2. **Install requirement libs**:
   ```bash
   pip install -r requirements.txt
   ```


   ```

## Results

The system achieved high classification accuracy, demonstrating its ability to generalize well even with significant appearance variations. The 1D-CNN classifier effectively captured discriminative features in the embedding space, enabling reliable identity recognition.


## Acknowledgments

- FaceNet: [https://github.com/davidsandberg/facenet](https://github.com/davidsandberg/facenet)
- Keras, TensorFlow
- Research inspiration from academic literature in face recognition and biometric authentication

