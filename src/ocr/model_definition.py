import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, img_channels, num_classes, rnn_hidden_size=512, rnn_num_layers=2, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(img_channels, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2,1), (2,1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2,1), (2,1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.rnn = nn.Sequential(
            nn.LayerNorm(512),
            nn.LSTM(512, rnn_hidden_size, rnn_num_layers, bidirectional=True, batch_first=False, dropout=dropout)
        )
        self.fc = nn.Linear(2 * rnn_hidden_size, num_classes)

    def forward(self, x):
        conv = self.features(x)         # [B, 512, 1, W']
        b, c, h, w = conv.size()
        assert h == 1, f"Conv features must have height=1, got {h}"
        conv = conv.squeeze(2)          # [B, 512, W]
        conv = conv.permute(2, 0, 1)    # [W, B, 512]
        rnn_out, _ = self.rnn(conv)     # [W, B, 2*rnn_hidden_size]
        logits = self.fc(rnn_out)       # [W, B, num_classes]
        return logits
