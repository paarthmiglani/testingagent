import pandas as pd
import cv2
from matplotlib import pyplot as plt

df = pd.read_csv("data/train_annotations.csv")
for idx, row in df.sample(10).iterrows():
    img = cv2.imread(f"data/images/{row['filename']}")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(row['text'])
    plt.show()
