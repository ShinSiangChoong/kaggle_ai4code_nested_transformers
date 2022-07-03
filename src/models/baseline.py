import torch
import torch.nn as nn
import torch.nn.functional as F
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


class Linear(nn.Linear):
    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.01) 
            

class Attention(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = Linear(dim, dim, bias=False)
        self.v = Linear(dim, 1, bias=False)

    def forward(self, keys: torch.Tensor, masks: torch.Tensor):
        weights = self.v(torch.tanh(self.W(keys)))
        weights.masked_fill_(masks, -6.5e4)
        weights = F.softmax(weights, dim=2)
        return torch.sum(weights * keys, dim=2)
    

class NotebookModel(nn.Module):
    def __init__(self, model_path):
        super(NotebookModel, self).__init__()
        self.tfm = AutoModel.from_pretrained(model_path)
        self.agg = Attention(768)       
        self.l0 = Linear(769, 256)
        self.l1 = Linear(256, 128)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.LeakyReLU()
        self.top = Linear(128, 1)

    def forward(self, ids, masks, fts, label_mask):
        bs, n_cells, max_len = ids.shape
        ids = ids.view(-1, max_len)
        masks = masks.view(-1, max_len)
        x = self.tfm(ids, masks)[0]
        x = x.view(bs, n_cells, max_len, x.shape[-1])
        x = self.agg(x, (1-masks).bool().view(bs, n_cells, max_len, -1))
        x = self.tfm(inputs_embeds=x, attention_mask=label_mask)[0]
        x = torch.cat((x, fts.unsqueeze(1).repeat(1, 128, 1)), 2)
        x = self.act(self.l0(x))
        x = self.dropout(x)
        x = self.act(self.l1(x))
        x = F.relu(self.top(x).squeeze(-1))
        return x