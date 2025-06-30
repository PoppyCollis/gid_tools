import torch.nn as nn
import torch.nn.functional as F

class ToolCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(ToolCNN, self).__init__()
        # Input: (1, 28, 28)
        self.conv1 = nn.Conv2d(1,   32, kernel_size=3, padding=1)  # → (32,28,28)
        self.conv2 = nn.Conv2d(32,  64, kernel_size=3, padding=1)  # → (64,14,14) after pool
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # → (128,7,7) after pool
        self.pool         = nn.MaxPool2d(2, 2)
        # channel-wise dropout for conv feature maps
        self.dropout_conv = nn.Dropout2d(0.25)
        # standard dropout for FC layers
        self.dropout_fc   = nn.Dropout(0.5)

        # After three poolings: 32→16→8→4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))         # 32→16
        x = self.pool(F.relu(self.conv2(x)))         # 16→8
        x = self.pool(F.relu(self.conv3(x)))         # 8→4
        x = self.dropout_conv(x)                     # channel-wise dropout

        x = x.view(x.size(0), -1)               # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)                  # standard dropout
        x = self.fc2(x)
        return x