import torch
import torch.nn as nn

class IMUModel(nn.Module):
    """
    組員 3 的 IMU 模型骨架 (收到組員程式碼後，將內部 Sequential 替換掉即可)
    """
    def __init__(self, in_channels=6, num_classes=3):
        super(IMUModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(32 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)