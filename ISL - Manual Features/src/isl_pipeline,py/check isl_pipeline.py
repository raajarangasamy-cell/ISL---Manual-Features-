import os
import numpy as np

class FinalDataPipeline:
    def __init__(self):
        pass

    def combine_all_features(self, features_dir, output_file_path):
        all_frames_data = []

        if os.path.exists(features_dir):
            sorted_files = sorted([f for f in os.listdir(features_dir) if f.endswith('.npy') and not f.endswith('final_sequence_data.npy')])

            for file_name in sorted_files:
                file_path = os.path.join(features_dir, file_name)
                try:
                    data = np.load(file_path)

                    all_frames_data.append(data)
                except Exception as e:
                    print(f"Error in {file_name}: {e}")
                    continue

        if len(all_frames_data) > 0:
            final_sequence = np.array(all_frames_data)
            np.save(output_file_path, final_sequence)
            print(f"[SUCCESS] Created final file at: {output_file_path}")
        else:
            print("[ERROR] No feature files found to combine!")


if __name__ == "__main__":
    features_directory = r"C:\Users\revas\PycharmProjects\ISL convert into text\src\video_processor.py\data\features"
    final_output_file = r"C:\Users\revas\PycharmProjects\ISL convert into text\src\video_processor.py\data\features\final_sequence_data.npy"

    pipeline = FinalDataPipeline()
    pipeline.combine_all_features(features_directory, final_output_file)