import torch
import torch.nn as nn

class FasterRCNNOutput(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.c_score = nn.Linear(in_channels, n_classes)
        self.bbx_pred = nn.Linear(in_channels, n_classes)
    
    def forward(self, x):
        x = x.flatten(start_dim=1)
        score = self.c_score(x)
        bx = self.bbx_pred(x)
        return score, bx

class FasterRCNN:
    pass