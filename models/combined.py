import torch
import torch.nn as nn
from models.gnn import FaceGNN
from models.pinn import MotionPINN

class FaceCursorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn = FaceGNN()
        self.pinn = MotionPINN()

    def forward(self, x, edge_index):
        pos = self.gnn(x, edge_index)
        smooth_pos = self.pinn(pos)
        return smooth_pos
