# Capturing Non-Manual Features of Indian Sign Language and Converting It Into Text

## Project Overview

This project aims to capture non-manual features of Indian Sign Language (ISL) such as facial expressions, eye movements, and head movements, and convert them into text. The system uses computer vision and deep learning techniques to process sign language videos and generate corresponding text output.

## Objectives

* Extract frames from ISL videos.
* Detect facial landmarks using MediaPipe Face Mesh.
* Track eye movements.
* Estimate head pose.
* Extract visual features using ResNet50.
* Combine extracted features into a sequence dataset.
* Convert recognized signs into text.

## Technologies Used

* Python
* OpenCV
* MediaPipe
* TensorFlow
* NumPy
* ResNet50

## Project Structure

* video_processor.py – Video frame extraction
* face_detector.py – Face landmark detection
* eye_detector.py – Eye tracking
* head_detector.py – Head pose estimation
* feature_extractor.py – ResNet50 feature extraction
* isl_pipeline.py – Feature combination pipeline
* text_converter.py – ISL text generation

## Current Output

The system successfully processes sign language frames and generates text output such as:

PREDICTED ISL SIGN TEXT : [ YES ]

## Future Enhancements

* Train a real machine learning model for sign classification.
* Support a larger ISL vocabulary.
* Improve non-manual feature integration.
* Develop a real-time ISL-to-text application.
