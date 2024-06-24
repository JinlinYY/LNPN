import torch.nn as nn
import torch.nn.functional as F
class MLP_Classifier(nn.Module):
    def __init__(self, feature_dim, output_size):
        super().__init__()
        self.linear1 = nn.Linear(feature_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.linear4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.classifier = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.linear3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.linear4(x)))
        x = self.dropout(x)
        out = self.classifier(x)
        return out
