import cv2
import os

# Current working directory check
print("Current Folder:", os.getcwd())

# Input and output folders
input_folder = "data/raw_videos"
output_folder = "data/processed"

# Create processed folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Check if input folder exists
if not os.path.exists(input_folder):
    print(f"Error: '{input_folder}' folder not found!")
    exit()

# Process all videos
for video_file in os.listdir(input_folder):

    # Process only mp4 files
    if not video_file.endswith(".mp4"):
        continue

    video_path = os.path.join(input_folder, video_file)

    print(f"Processing: {video_file}")

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    video_name = os.path.splitext(video_file)[0]

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_filename = f"{video_name}_frame_{frame_count}.jpg"
        frame_path = os.path.join(output_folder, frame_filename)

        cv2.imwrite(frame_path, frame)

        # Skip 30 frames
        frame_count += 30
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

    cap.release()

    print(f"Finished: {video_file}")

print("\nAll videos processed successfully!")