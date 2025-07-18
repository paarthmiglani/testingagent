import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ocr.dataset import OCRDataset, ocr_collate_fn
from src.ocr.model_definition import CRNN
from src.ocr.utils import load_char_list

# --- CER/WER functions ---
def cer(ref, hyp):
    import editdistance
    return editdistance.eval(ref, hyp) / max(1, len(ref))

def wer(ref, hyp):
    ref_words = ref.split()
    hyp_words = hyp.split()
    import editdistance
    return editdistance.eval(ref_words, hyp_words) / max(1, len(ref_words))

def decode_predictions(log_probs, input_lens, idx_to_char):
    # log_probs: (seq_len, batch, num_classes)
    pred_indices = log_probs.argmax(2).cpu().numpy().T  # (batch, seq_len)
    texts = []
    for inds, L in zip(pred_indices, input_lens.cpu().numpy()):
        seq = []
        prev = None
        for idx in inds[:L]:
            if idx != 0 and idx != prev:  # 0 is BLANK
                seq.append(idx_to_char[idx])
            prev = idx
        texts.append("".join(seq))
    return texts

def evaluate(model, val_loader, device, idx_to_char):
    model.eval()
    total_loss = 0.0
    total_cer = 0.0
    total_wer = 0.0
    n_samples = 0

    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for images, labels_concat, label_strs, label_lens in pbar:
            images = images.to(device)
            labels_concat = labels_concat.to(device)
            label_lens = label_lens.to(device)

            log_probs = model(images)  # [seq_len, batch, num_classes]
            log_probs = log_probs.log_softmax(2)
            seq_len = log_probs.size(0)
            batch_size = log_probs.size(1)
            input_lens = torch.full((batch_size,), seq_len, dtype=torch.long, device=log_probs.device)

            loss = ctc_loss(log_probs, labels_concat, input_lens, label_lens)
            total_loss += loss.item()

            pred_texts = decode_predictions(log_probs, input_lens, idx_to_char)
            for pred, gt in zip(pred_texts, label_strs):
                total_cer += cer(gt, pred)
                total_wer += wer(gt, pred)
            n_samples += images.size(0)

    avg_loss = total_loss / len(val_loader)
    avg_cer = total_cer / n_samples
    avg_wer = total_wer / n_samples
    return avg_loss, avg_cer, avg_wer

def train():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config['model']
    train_cfg = config['training']
    pre_cfg = config.get('preprocessing', {})

    char_list_path = train_cfg['char_list_path']
    char_list = load_char_list(char_list_path)
    idx_to_char = {i: c for i, c in enumerate(char_list)}
    print(f"Loaded {len(char_list)} characters from {char_list_path}.")
    char_to_idx = {c: i for i, c in enumerate(char_list)}

    train_dataset = OCRDataset(
        annotations_file=train_cfg['annotations_file'],
        img_dir=train_cfg['dataset_path'],
        char_to_idx=char_to_idx
    )
    val_dataset = OCRDataset(
        annotations_file=train_cfg['validation_annotations_file'],
        img_dir=train_cfg['validation_dataset_path'],
        char_to_idx=char_to_idx
    )

    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True,
                              num_workers=2, collate_fn=ocr_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False,
                            num_workers=2, collate_fn=ocr_collate_fn)

    model = CRNN(
        img_channels=1,
        num_classes=len(char_list),
        rnn_hidden_size=model_cfg.get('rnn_hidden_size', 512),
        rnn_num_layers=model_cfg.get('num_rnn_layers', 2)
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['learning_rate'],
                                 weight_decay=train_cfg['weight_decay'])
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    epochs = train_cfg['epochs']
    out_dir = train_cfg.get('output_dir', './models/ocr/')
    os.makedirs(out_dir, exist_ok=True)

    print("\nStarting OCR training...\n")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (images, labels_concat, _, label_lens) in pbar:
            images = images.to(device)
            labels_concat = labels_concat.to(device)
            label_lens = label_lens.to(device)

            logits = model(images)  # (seq_len, batch, num_classes)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)  # along num_classes
            seq_len = log_probs.size(0)
            input_lens = torch.full(size=(log_probs.size(1),), fill_value=seq_len, dtype=torch.long).to(device)

            loss = ctc_loss(log_probs, labels_concat, input_lens, label_lens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Batch": f"{batch_idx}/{len(train_loader)}"})

        avg_loss = running_loss / len(train_loader)
        print(f"\nEnd of Epoch {epoch+1}, Average Training Loss: {avg_loss:.4f}")

        # Validation & Metrics
        val_loss, val_cer, val_wer = evaluate(model, val_loader, device, idx_to_char)
        print(f"Validation Results Epoch {epoch+1}: Loss={val_loss:.4f}, CER={val_cer:.4f}, WER={val_wer:.4f}")

        torch.save(model.state_dict(), os.path.join(out_dir, f"crnn_newdef_{epoch+1}.pth"))

if __name__ == "__main__":
    train()
