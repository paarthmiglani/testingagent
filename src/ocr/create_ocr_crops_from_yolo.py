import os
import cv2
import csv

# Paths to your folders
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]

images_dir = BASE_DIR / "data" / "images"
labels_dir = BASE_DIR / "data" / "labels"
source_anno_csv = BASE_DIR / "data" / "train_annotations.csv"
crops_dir = BASE_DIR / "data" / "ocr_crops"
ANNOT_OUT = BASE_DIR / "data" / "crop_annotations.csv"

os.makedirs(crops_dir, exist_ok=True)

# Step 1: Build mapping from image filename to list of text lines (ground truth)
text_map = {}
with open(source_anno_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        fname, txt = row['filename'], row['text']
        text_map.setdefault(fname, []).append(txt)

# Step 2: For each image, process its YOLO label file and crop each box
out_rows = []
for img_name in os.listdir(images_dir):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(images_dir, img_name)
    label_path = os.path.join(labels_dir, img_name.rsplit('.', 1)[0] + ".txt")

    if not os.path.exists(label_path):
        continue

    # Read original image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: could not read image {img_name}")
        continue
    h, w = img.shape[:2]

    # Get all text lines for this image, or fallback to 'unknown'
    img_texts = text_map.get(img_name, [])

    # Read YOLO labels (each line: class_id x_center y_center width height)
    with open(label_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, x_c, y_c, bw, bh = map(float, parts)
        # Convert YOLO normalized to pixel coordinates
        x1 = int((x_c - bw/2) * w)
        y1 = int((y_c - bh/2) * h)
        x2 = int((x_c + bw/2) * w)
        y2 = int((y_c + bh/2) * h)
        # Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        crop = img[y1:y2, x1:x2]

        # Save crop
        crop_name = f"{img_name.rsplit('.',1)[0]}_crop_{i}.jpg"
        crop_path = os.path.join(crops_dir, crop_name)
        if crop.size == 0:
            print(f"Warning: Empty crop for {img_name} at box {i}: [{x1},{y1},{x2},{y2}]")
            continue
        cv2.imwrite(crop_path, crop)

        # Assign text: use annotation if available, else 'unknown'
        assigned_text = img_texts[i] if i < len(img_texts) else 'unknown'
        out_rows.append([crop_name, assigned_text])

# Step 3: Save the crops annotation CSV
with open(ANNOT_OUT, "w", encoding="utf-8", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "text"])
    writer.writerows(out_rows)

print(f"Done! Created {len(out_rows)} crops and annotation file at {ANNOT_OUT}")
