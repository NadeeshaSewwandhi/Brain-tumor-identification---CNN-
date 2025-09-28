# Brain-tumor-identification---CNN
Brain Tumor Detection System

This project is a deep learning–based web application that detects brain tumors from MRI scans. A Convolutional Neural Network (CNN) model is trained to classify images into two categories:

 1. Normal brain scan (No tumor detected)
 2. Abnormal brain scan (Possible brain tumor identified)

The trained model is integrated with a Flask web application, allowing users to upload MRI images through a browser and instantly receive predictions.

Features :

1. Image Upload: Users can upload brain MRI images via the web interface.
2. Automated Prediction: The model predicts whether the image shows a tumor or not.
3. User-Friendly Output: Returns descriptive results instead of just numeric labels.
4. Flask Integration: Provides an accessible web-based solution.
5. Reusable Model: CNN model saved in .h5 format can be reused or improved with more data.

Workflow :

1. Dataset Preparation
MRI images are categorized into two folders: yes (tumor present) and no (no tumor).
Images are resized to 64x64 pixels for uniform input size.

2. Model Training (Python script)
A CNN model is built using Keras.
Data is normalized and split into training/testing sets.
Model is trained for 10 epochs and saved as BrainTumor10EpochsCategorical.h5.

3. Prediction Script
A test script reads an image, preprocesses it, and predicts the class using the trained model.

4. Flask Web App
Users access the homepage (index.html).
Upload an MRI image through the web interface.
The backend preprocesses the image, runs prediction, and returns the result.

Output
>> If class 0: "Normal brain scan - No tumor Detected"
>> If class 1: "Abnormal findings - Possible brain tumor identified"

Technologies Used :

1. Programming Languages: Python
2. Deep Learning Frameworks: TensorFlow, Keras
3. Image Processing: OpenCV, PIL (Pillow), NumPy
4. Web Framework: Flask
5. Frontend: HTML (index.html template), optional JavaScript
6. Utilities: Werkzeug (for secure file uploads), OS module

Learning Outcomes:

By building this project, the following concepts and skills were gained:

1. Understanding Convolutional Neural Networks (CNNs) for image classification.
2. Preprocessing images (resizing, normalization, one-hot encoding).
3. Training, evaluating, and saving models using Keras.
4. Handling user-uploaded files securely in Flask.
5. Integrating machine learning models with web applications.
6. Improving problem-solving and debugging skills in ML workflows.
