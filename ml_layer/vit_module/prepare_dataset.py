"""
Downloads Food-101 (non-streaming - uses HF's parallelized download/cache),
then filters to our 10 target classes and saves a small image subset locally.
"""

import os
from datasets import load_dataset

TARGET_CLASSES = [
    "pizza", "hamburger", "samosa", "fried_rice", "ice_cream",
    "french_fries", "chicken_curry", "omelette", "spring_rolls", "ramen",
]

TRAIN_PER_CLASS = 40
VAL_PER_CLASS = 10

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def prepare():
    print("Downloading Food-101 metadata + images (cached after first run)...")
    dataset = load_dataset("ethz/food101", split="train")
    print(f"Full dataset loaded: {len(dataset)} images total.")

    label_names = dataset.features["label"].names
    target_indices = {label_names.index(cls): cls for cls in TARGET_CLASSES if cls in label_names}
    print(f"Matched {len(target_indices)} of {len(TARGET_CLASSES)} target classes: {list(target_indices.values())}")

    print("Filtering dataset to target classes (parallelized)...")
    filtered = dataset.filter(lambda ex: ex["label"] in target_indices, num_proc=4)
    print(f"Filtered dataset size: {len(filtered)} images.")

    counts = {cls: 0 for cls in target_indices.values()}
    needed_train = TRAIN_PER_CLASS
    needed_val = VAL_PER_CLASS

    for split_name in ["train", "val"]:
        for cls in target_indices.values():
            os.makedirs(os.path.join(OUTPUT_DIR, split_name, cls), exist_ok=True)

    print("Saving images to disk...")
    saved = 0
    for example in filtered:
        cls_name = target_indices[example["label"]]

        if counts[cls_name] < needed_train:
            split_name = "train"
            idx = counts[cls_name]
        elif counts[cls_name] < needed_train + needed_val:
            split_name = "val"
            idx = counts[cls_name] - needed_train
        else:
            continue

        img_path = os.path.join(OUTPUT_DIR, split_name, cls_name, f"{idx}.jpg")
        example["image"].convert("RGB").save(img_path)
        counts[cls_name] += 1
        saved += 1

        if saved % 50 == 0:
            print(f"  Saved {saved} images so far... ({counts})")

        if all(v >= needed_train + needed_val for v in counts.values()):
            break

    print("Done. Final counts per class:", counts)
    print("Dataset preparation complete. Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    prepare()
