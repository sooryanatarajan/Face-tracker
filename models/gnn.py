import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class FaceGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 64)
        self.conv2 = GCNConv(64, 128)
        self.fc = nn.Linear(128, 2)  # predict x,y

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()

        nose_feat = x[1]  # nose node
        return self.fc(nose_feat)
