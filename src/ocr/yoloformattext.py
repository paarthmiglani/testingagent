import os
import glob

def quad_to_yolo_bbox(coords, img_w, img_h):
    # coords: [x1,y1,x2,y2,x3,y3,x4,y4]
    xs = [float(coords[i]) for i in range(0,8,2)]
    ys = [float(coords[i]) for i in range(1,8,2)]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    # YOLO expects: x_center, y_center, width, height (normalized)
    x_c = (x_min + x_max) / 2.0 / img_w
    y_c = (y_min + y_max) / 2.0 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return x_c, y_c, w, h

# ---- CHANGE THESE -----
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]

IMAGES_DIR = BASE_DIR / "data" / "images"
LABELS_IN_DIR = BASE_DIR / "data" / "imagestext"
LABELS_OUT_DIR = BASE_DIR / "data" / "labels"
os.makedirs(LABELS_OUT_DIR, exist_ok=True)

from PIL import Image

for txtfile in glob.glob(os.path.join(LABELS_IN_DIR, "gt_img_*.txt")):
    img_name = os.path.basename(txtfile).replace("gt_", "").replace(".txt", ".jpg")
    img_path = os.path.join(IMAGES_DIR, img_name)
    if not os.path.exists(img_path):
        continue
    img_w, img_h = Image.open(img_path).size
    label_lines = []
    with open(txtfile, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue
            coords = parts[:8]
            # lang, text = parts[8], parts[9]
            x_c, y_c, w, h = quad_to_yolo_bbox(coords, img_w, img_h)
            line_out = f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
            label_lines.append(line_out)
    out_txt = os.path.join(LABELS_OUT_DIR, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))
    with open(out_txt, "w") as fout:
        fout.write("\n".join(label_lines) + "\n")

print(f"Done! Created YOLO label files in {LABELS_OUT_DIR}")
