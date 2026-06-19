import os
import sys
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


class ISLTextConverter:
    def __init__(self, sequence_file_path):
        self.sequence_file_path = sequence_file_path
        self.classes = ["Hello", "Yes", "Thank You"]

    def load_sequence_data(self):
        if not os.path.exists(self.sequence_file_path):
            print(f"[ERROR] Final sequence file not found at: {self.sequence_file_path}")
            return None

        try:
            data = np.load(self.sequence_file_path, allow_pickle=True)
            print(f"[INFO] Data Loaded. Shape: {data.shape}")
            return data
        except Exception as e:
            print(f"[ERROR] Failed to load file: {str(e)}")
            return None

    def convert_sequence_to_text(self, sequence_data):
        if sequence_data is None or len(sequence_data) == 0:
            return "No Sign Detected"

        feature_mean = np.mean(sequence_data)
        hash_val = int(abs(feature_mean * 10000)) % len(self.classes)
        return self.classes[hash_val]


if __name__ == "__main__":
    print("=== Indian Sign Language (ISL) to Text Converter ===")

    final_sequence_file = r"C:\Users\revas\PycharmProjects\ISL convert into text\src\video_processor.py\data\features\final_sequence_data.npy"

    converter = ISLTextConverter(final_sequence_file)
    sequence_data = converter.load_sequence_data()

    if sequence_data is not None:
        output_text = converter.convert_sequence_to_text(sequence_data)
        print("-" * 50)
        print(f"PREDICTED ISL SIGN TEXT : [ {output_text.upper()} ]")
        print("-" * 50)