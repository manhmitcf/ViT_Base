import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class FishClassifier(nn.Module):
    def __init__(self, num_classes=3): 
        super(FishClassifier, self).__init__()
        # Load the ViT Base model with pre-trained weights
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        
        # Replace the classifier head with a new one for the specific number of classes
        self.vit.heads = nn.Linear(self.vit.heads.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)