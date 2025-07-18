import os
import glob

# Define paths
IMAGE_DIR = "/Users/paarthmiglani/PycharmProjects/manualagent/yolo/images"
GT_DIR = "/Users/paarthmiglani/PycharmProjects/manualagent/yolo/text"
OUT_LABEL_DIR = "/Users/paarthmiglani/PycharmProjects/manualagent/yolo/labels"

os.makedirs(OUT_LABEL_DIR, exist_ok=True)

def polygon_to_yolo_bbox(polygon, img_w, img_h):
    # polygon: [x1,y1,x2,y2,x3,y3,x4,y4]
    xs = polygon[0::2]
    ys = polygon[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    # YOLO normalized
    x_c = (x_min + x_max) / 2.0 / img_w
    y_c = (y_min + y_max) / 2.0 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return [0, x_c, y_c, w, h]  # class 0 for text

for img_path in glob.glob(os.path.join(IMAGE_DIR, "*")):
    img_name = os.path.basename(img_path)
    img_base, ext = os.path.splitext(img_name)
    gt_file = os.path.join(GT_DIR, f"gt_{img_base}.txt")
    out_label = os.path.join(OUT_LABEL_DIR, f"{img_base}.txt")
    label_lines = []
    if os.path.exists(gt_file):
        import cv2
        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot read {img_path}")
            continue
        img_h, img_w = img.shape[:2]
        with open(gt_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 10: continue
                try:
                    polygon = [float(x) for x in parts[:8]]
                except:
                    continue
                yolo_box = polygon_to_yolo_bbox(polygon, img_w, img_h)
                yolo_str = " ".join([str(round(x, 6)) for x in yolo_box])
                label_lines.append(yolo_str)
    # Always create a file, even if empty (no text lines)
    with open(out_label, "w", encoding="utf-8") as f:
        if label_lines:
            f.write("\n".join(label_lines) + "\n")
print(f"Done! Created YOLO label files in {OUT_LABEL_DIR}")
