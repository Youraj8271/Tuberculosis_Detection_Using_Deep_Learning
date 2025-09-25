# src/model.py
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

def get_model(num_classes: int = 2, pretrained: bool = True):
    """
    Returns a ResNet18 model for image classification.
    - Replaces the final fully connected (FC) layer with a new one for `num_classes`.
    - Uses new torchvision API with 'weights' instead of deprecated 'pretrained'.
    """

    # Use updated API: 'weights' instead of 'pretrained'
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model
