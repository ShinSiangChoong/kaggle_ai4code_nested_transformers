dependencies = ['torch', 'transformers']
from typing import Optional
from torch import nn
import torch

# NEW STRUCTURE
#from src.models.naive.vit import *
#from src.models.naive.resnet import *
#from src.models.naive.convnext import *


def dummy(weights_file: Optional[str] = None) -> nn.Module:
    """
    Dummy model

    Just so other implementations can use this as a bootstrapping point
    Args:
        weights_file (str, optional): If specified, load state dictionary from this file.
    """
    model = torch.hub.load('pytorch/vision:v0.10.0', model="resnet34", pretrained=False)
    model.fc = nn.Linear(in_features=512, out_features=80000, bias=True)

    if weights_file:
        print(f"Loading state from {weights_file}")
        model.load_state_dict(torch.load(weights_file))

    return model
