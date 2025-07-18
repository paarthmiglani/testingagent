import os
import torch
import pandas as pd
import cv2
import numpy as np

def preprocess_image(image_path, target_size=(128, 32), binarize=False):
    """Load and preprocess a grayscale image for OCR."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    if binarize:
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Shape: (1, H, W)
    return img

class OCRDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, char_to_idx, transform=None, binarize=False):
        """
        annotations_file: Path to CSV with 'filename' and 'text'
        img_dir: Directory with images
        char_to_idx: Dict mapping chars to indices
        transform: Optional, additional augmentations
        binarize: Optional, apply binarization to images
        """
        self.data = pd.read_csv(annotations_file)
        self.data = self.data.dropna(subset=['filename', 'text']).reset_index(drop=True)
        self.img_dir = img_dir
        self.char_to_idx = char_to_idx
        self.transform = transform
        self.binarize = binarize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_filename = row['filename']
        text = str(row['text']) if not pd.isna(row['text']) else ""

        img_path = os.path.join(self.img_dir, img_filename)
        image = preprocess_image(img_path, binarize=self.binarize)

        if self.transform:
            # torchvision-style transforms expect PIL, but you can adapt as needed
            image = self.transform(image)

        # Convert label to indices
        label = torch.tensor([self.char_to_idx.get(c, 0) for c in text], dtype=torch.long)
        # 0 is usually the BLANK index for CTC (make sure your char_to_idx maps blank/unknown to 0!)

        return torch.tensor(image, dtype=torch.float32), label, text

def ocr_collate_fn(batch):
    """
    Collate function for OCR dataset to handle variable-length sequences.
    Pads images to max width in batch and concatenates labels for CTC.
    Returns:
        images: Tensor [B, 1, H, max_W]
        labels_concat: Tensor [sum(label_lens)]
        texts: list of str (GT labels)
        label_lens: Tensor [B]
    """
    images, labels, texts = zip(*batch)
    # Pad images to max width in the batch
    heights = [img.shape[1] for img in images]
    widths = [img.shape[2] for img in images]
    max_w = max(widths)
    padded_imgs = []
    for img in images:
        pad_width = max_w - img.shape[2]
        if pad_width > 0:
            img = np.pad(img, ((0,0), (0,0), (0, pad_width)), mode='constant', constant_values=0)
        padded_imgs.append(img)
    images_tensor = torch.tensor(np.stack(padded_imgs), dtype=torch.float32)

    # Concatenate labels for CTC loss
    label_lens = torch.tensor([len(lab) for lab in labels], dtype=torch.long)
    if len(labels) and labels[0].ndim == 1:
        labels_concat = torch.cat(labels, dim=0)
    else:
        labels_concat = torch.tensor([], dtype=torch.long)

    return images_tensor, labels_concat, list(texts), label_lens
