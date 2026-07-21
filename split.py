import os
import shutil
import random

SOURCE = "dataset_frames"
DEST = "dataset_split"

CLASSES = ["positive", "negative"]

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    for cls in CLASSES:
        files = os.listdir(os.path.join(SOURCE, cls))
        files = [f for f in files if f.endswith(".jpg")]

        random.shuffle(files)

        train_end = int(len(files) * TRAIN_SPLIT)
        val_end = train_end + int(len(files) * VAL_SPLIT)

        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        for split_name, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            split_path = os.path.join(DEST, split_name, cls)
            ensure_dir(split_path)

            for f in split_files:
                src_path = os.path.join(SOURCE, cls, f)
                dst_path = os.path.join(split_path, f)
                shutil.copy(src_path, dst_path)

        print(f"{cls}: Train={len(train_files)} Val={len(val_files)} Test={len(test_files)}")

    print("DONE. Split created in 'dataset_split/' folder.")

if __name__ == "__main__":
    main()
