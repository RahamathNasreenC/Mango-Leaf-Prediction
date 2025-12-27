import os
import random
import shutil

def split_data(source_dir, train_dir, test_dir, split_ratio=0.8):
    if os.path.exists(train_dir): shutil.rmtree(train_dir)
    if os.path.exists(test_dir): shutil.rmtree(test_dir)

    for category in os.listdir(source_dir):
        src_path = os.path.join(source_dir, category)
        files = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]
        random.shuffle(files)

        split_point = int(len(files) * split_ratio)
        train_files = files[:split_point]
        test_files = files[split_point:]

        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(test_dir, category), exist_ok=True)

        for f in train_files:
            shutil.copy(os.path.join(src_path, f), os.path.join(train_dir, category, f))
        for f in test_files:
            shutil.copy(os.path.join(src_path, f), os.path.join(test_dir, category, f))

    print("✅ Dataset split complete!")

split_data("all_images", "data/train", "data/test")
