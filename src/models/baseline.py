import torch
import torch.nn as nn
from transformers import AutoModel


class MarkdownModel(nn.Module):
    def __init__(self, model_path):
        super(MarkdownModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.top = nn.Linear(769, 1)

    def forward(self, ids, mask, fts):
        x = self.model(ids, mask)[0]
        x = torch.cat((x, fts.unsqueeze(1).repeat(1, 512, 1)), 2)
        x = self.top(x)
        return x.squeeze(-1)