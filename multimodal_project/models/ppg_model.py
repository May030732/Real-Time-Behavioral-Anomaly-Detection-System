import torch
import torch.nn as nn

class StressNN(nn.Module):
    """
    PPG/HRV 心率壓力檢測模型 (19 個特徵維度 -> 3 個類別)
    """
    def __init__(self, input_dim=19, num_classes=3):
        super(StressNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)