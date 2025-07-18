import numpy as np

def ctc_decode_predictions(predictions, char_list, blank_idx=0):
    pred_idxs = np.argmax(predictions, axis=-1)
    decoded = []
    prev_idx = -1
    for idx in pred_idxs:
        if idx != blank_idx and idx != prev_idx:
            decoded.append(char_list[idx])
        prev_idx = idx
    return ''.join(decoded)
