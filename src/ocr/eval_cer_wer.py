from jiwer import cer, wer

def calculate_cer(reference, hypothesis):
    return cer(reference, hypothesis)

def calculate_wer(reference, hypothesis):
    return wer(reference, hypothesis)
