import albumentations as A

def get_augmentations():
    return A.Compose([
        A.Rotate(limit=5, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussianBlur(p=0.2),
    ])
