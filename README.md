# Indic-OCR Chatbot Demo

This project combines simple OCR and NLP components to explore Indian cultural texts.


## Directory structure

```
configs/         # YAML config files
streamlit_app/   # Streamlit interface
src/             # Python source code
    nlp/         # NLP utilities and chatbot agent
    ocr/         # OCR model, dataset and helpers
scripts/         # Helper scripts
runs/            # Training outputs (models and logs)
data/            # Data files (images, annotations, NLP corpora)
```

`configs` hold the paths for training data and character lists. Update these YAML files if your data is stored elsewhere.

## Running the Streamlit app

Install the required packages (e.g. `streamlit`, `pandas`, `easyocr`, `torch`):

```bash
pip install -r requirements.txt  # or manually install the dependencies
```

Then launch the UI:

```bash
streamlit run streamlit_app/app.py
```

The app loads transliteration files and knowledge CSVs from the `data/` directory. Adjust `streamlit_app/app.py` if your files live elsewhere.

## Training the OCR model

The OCR training script reads its configuration from a YAML file. Example:

```bash
python src/ocr/train.py --config configs/ocr.yaml
```

Paths for images, annotations and output models are defined inside the config. For cropped text training use `configs/ocr_crops.yaml`.

YOLO detection training can be run with the Ultralytics CLI using `configs/yolo_data.yaml` as the data description and `yolov8n.pt` as the starting weights.

---
Data files referenced in the configs are expected under the repository's `data/` directory by default:

```
data/images/                  # training images
data/validation_images/       # validation images
data/train_annotations.csv    # annotation CSV
...
```

Modify the paths in `configs/*.yaml` to point to your own datasets if they are stored in a different location.

