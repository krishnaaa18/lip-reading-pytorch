import torch
import torch.nn as nn

class LipReadingModel(nn.Module):
    def __init__(self, num_classes, hidden_size=64):
        super(LipReadingModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))  # final size: [batch, 64, 1, 1]

        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=1, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: [batch, seq_len, 1, 64, 64]
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)  # merge batch and time

        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)  # [b*t, 64, 1, 1]
        x = x.view(b, t, -1)  # [batch, time, 64]

        x, _ = self.lstm(x)  # [batch, time, hidden]
        x = x[:, -1, :]  # take last frame's output

        out = self.fc(x)
        return out
