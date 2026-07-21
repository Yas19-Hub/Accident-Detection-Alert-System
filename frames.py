import os
import cv2

# CHANGE this to the path where your dataset folder is
DATASET_ROOT = r"dataset"

# Mapping from "class name" to subfolder
CLASS_DIRS = {
    "positive": "Positive_Vidoes",   # use exact folder name from your disk
    "negative": "Negative_Videos"
}

# Where we will save extracted frames as images
OUTPUT_ROOT = r"dataset_frames"  # this will be created

# Extract 1 frame every N frames (tune this)
FRAME_SKIP = 10  # e.g. take every 10th frame


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def extract_frames_for_class(label_name, subfolder_name):
    input_dir = os.path.join(DATASET_ROOT, subfolder_name)
    output_dir = os.path.join(OUTPUT_ROOT, label_name)
    ensure_dir(output_dir)

    video_extensions = (".mp4", ".avi", ".mov", ".mkv")

    video_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(video_extensions)]

    print(f"[{label_name}] Found {len(video_files)} videos in {input_dir}")

    for vid_idx, video_file in enumerate(video_files):
        video_path = os.path.join(input_dir, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"  !! Could not open video: {video_path}")
            continue

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # end of video

            # Only save every FRAME_SKIP-th frame
            if frame_count % FRAME_SKIP == 0:
                # Build filename: positive_0001_frame_000123.jpg
                frame_filename = f"{label_name}_{vid_idx:04d}_frame_{frame_count:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)

                # Optional: resize to fixed size now (e.g. 224x224)
                frame_resized = cv2.resize(frame, (224, 224))

                cv2.imwrite(frame_path, frame_resized)
                saved_count += 1

            frame_count += 1

        cap.release()
        print(f"  Processed {video_file}: total frames={frame_count}, saved={saved_count}")


def main():
    for label_name, subfolder_name in CLASS_DIRS.items():
        extract_frames_for_class(label_name, subfolder_name)

    print("DONE. Frames saved under:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
