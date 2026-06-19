# Project Architecture

## System Flow

Raw ISL Videos
↓
Frame Extraction (video_processor.py)
↓
Processed Frames
↓
Feature Extraction (feature_extractor.py)
↓
Visual Feature Files (.npy)
↓
Feature Combination (isl_pipeline.py)
↓
final_sequence_data.npy
↓
Text Conversion (text_converter.py)
↓
Predicted ISL Text

## Non-Manual Feature Modules

### Face Detector

* Detects facial landmarks using MediaPipe Face Mesh.
* Captures facial expressions.

### Eye Detector

* Tracks left and right eye landmarks.
* Captures eye movement information.

### Head Detector

* Estimates head pose.
* Extracts Pitch, Yaw and Roll values.

### Feature Extractor

* Uses ResNet50 pretrained model.
* Extracts 2048-dimensional visual features.

### ISL Pipeline

* Combines extracted features.
* Generates final sequence dataset.

### Text Converter

* Loads final sequence data.
* Converts extracted features into text output.

## Technologies Used

* Python
* OpenCV
* MediaPipe
* TensorFlow
* NumPy
* ResNet50

## Output

Example Output:

PREDICTED ISL SIGN TEXT : [ YES ]
