import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class FishClassifier(nn.Module):
    def __init__(self, num_classes=3): 
        super(FishClassifier, self).__init__()
        # Load the ViT Base model with pre-trained weights
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        
        # Access the last layer of the Sequential object to get in_features
        in_features = self.vit.heads[-1].in_features
        
        # Replace the classifier head with a new one for the specific number of classes
        self.vit.heads = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vit(x)