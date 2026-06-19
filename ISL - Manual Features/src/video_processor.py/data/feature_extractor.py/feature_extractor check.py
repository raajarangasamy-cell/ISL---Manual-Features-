import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


class FeatureExtractor:
    def __init__(self):
        self.model = ResNet50(
            weights="imagenet",
            include_top=False,
            pooling="avg"
        )

    def extract_features(self, frame):
        resized_frame = cv2.resize(frame, (224, 224))
        img_array = np.expand_dims(resized_frame, axis=0)
        preprocessed_img = preprocess_input(img_array)
        features = self.model.predict(preprocessed_img, verbose=0)
        return features.flatten()


if __name__ == "__main__":
    input_folder = r"C:\Users\revas\PycharmProjects\ISL convert into text\src\video_processor.py\data\processed"
    output_folder = r"C:\Users\revas\PycharmProjects\ISL convert into text\src\video_processor.py\data\features"

    os.makedirs(output_folder, exist_ok=True)
    extractor = FeatureExtractor()
    print("Input Folder:", input_folder)
    print("Files:", os.listdir(input_folder) if os.path.exists(input_folder) else "Folder Not Found")
    if os.path.exists(input_folder) and len(os.listdir(input_folder)) > 0:
        for img_name in os.listdir(input_folder):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(input_folder, img_name)
                frame = cv2.imread(img_path)

                if frame is not None:
                    features = extractor.extract_features(frame)
                    output_file_name = img_name.split('.')[0] + "_features.npy"
                    np.save(os.path.join(output_folder, output_file_name), features)
                    print(f"Saved: {output_file_name}")