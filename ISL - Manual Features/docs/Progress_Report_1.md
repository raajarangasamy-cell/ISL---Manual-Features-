# Progress Report 1

## Project Title

Capturing Non-Manual Features of Indian Sign Language and Converting It Into Text

## Objective

To develop a system that captures non-manual features of Indian Sign Language such as facial expressions, eye movements, and head movements, and converts them into meaningful text output.

## Work Completed

### Project Setup

* Created project structure.
* Configured Python environment.
* Installed required libraries.

### Data Processing

* Extracted frames from ISL videos.
* Organized processed image data.

### Feature Extraction

* Implemented ResNet50-based feature extraction.
* Generated feature files in .npy format.

### Non-Manual Feature Modules

* Developed face landmark detection module.
* Developed eye tracking module.
* Developed head pose estimation module.

### Pipeline Development

* Combined extracted features.
* Generated final sequence data file.

### Text Conversion

* Implemented ISL text conversion module.
* Successfully generated text output from processed data.

## Current Result

The system successfully processes sign language video frames and generates predicted text output.

Example:

PREDICTED ISL SIGN TEXT : [ YES ]

## Next Phase

* Improve prediction accuracy.
* Integrate all non-manual features into a unified model.
* Train a machine learning classifier.
* Support additional ISL signs.
