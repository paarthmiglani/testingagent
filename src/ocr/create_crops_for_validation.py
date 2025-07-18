import os
import cv2
import csv

# --- UPDATE THESE FOR VALIDATION SPLIT ---
images_dir = "/Users/paarthmiglani/PycharmProjects/manualagent/validation_images"
labels_dir = "/Users/paarthmiglani/PycharmProjects/manualagent/validation_labels"
source_anno_csv = "/Users/paarthmiglani/PycharmProjects/testingagent/data/val_annotations.csv"
crops_dir = "/Users/paarthmiglani/PycharmProjects/manualagent/ocr_crops_val"
ANNOT_OUT = "/Users/paarthmiglani/PycharmProjects/testingagent/data/crop_annotations_val.csv"

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

    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: could not read image {img_name}")
        continue
    h, w = img.shape[:2]

    img_texts = text_map.get(img_name, [])

    with open(label_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, x_c, y_c, bw, bh = map(float, parts)
        # YOLO format: normalized
        x1 = int((x_c - bw/2) * w)
        y1 = int((y_c - bh/2) * h)
        x2 = int((x_c + bw/2) * w)
        y2 = int((y_c + bh/2) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = img[y1:y2, x1:x2]

        crop_name = f"{img_name.rsplit('.',1)[0]}_crop_{i}.jpg"
        crop_path = os.path.join(crops_dir, crop_name)
        if crop.size == 0:
            print(f"Warning: Empty crop for {img_name} at box {i}: [{x1},{y1},{x2},{y2}]")
            continue
        cv2.imwrite(crop_path, crop)

        assigned_text = img_texts[i] if i < len(img_texts) else 'unknown'
        out_rows.append([crop_name, assigned_text])

# Step 3: Save the crops annotation CSV
with open(ANNOT_OUT, "w", encoding="utf-8", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "text"])
    writer.writerows(out_rows)

print(f"Done! Created {len(out_rows)} validation crops and annotation file at {ANNOT_OUT}")
