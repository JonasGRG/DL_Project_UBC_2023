import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    '''
    Multi-class Focal Loss
    '''
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if weight is not None:
            self.register_buffer('weight', weight)  # Register weight as a buffer, not a parameter
        else:
            self.weight = None  # Handle the case when weight is None

    def forward(self, input, target):
        """
        input: [N, C], float32
        target: [N, ], int64
        """
        if input.device != self.weight.device:
            self.weight = self.weight.to(input.device)
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss