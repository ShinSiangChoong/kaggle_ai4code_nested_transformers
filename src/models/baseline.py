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
        self.l0 = Linear(1537, 512)
        self.l1 = Linear(512, 128)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.LeakyReLU()
        self.top = Linear(128, 1)
        self.pad_logits = torch.Tensor([-6.5e-4])

    def forward(self, ids, cell_masks, nb_masks, label_mask, pos, start, fts):
        bs, n_cells, max_len = ids.shape
        ids = ids.view(-1, max_len)
        cell_masks = cell_masks.view(-1, max_len)
        
        # cell transformer
        x = self.tfm(ids, cell_masks)[0]
        x = x.view(bs, n_cells, max_len, x.shape[-1])
        x = self.agg(x, (1-cell_masks).bool().view(bs, n_cells, max_len, -1))
        
        # add start embs
        x = torch.cat((self.tfm.embeddings(start), x), dim=1)
        # notebook transformer
        x = self.tfm(
            inputs_embeds=x, 
            attention_mask=nb_masks, 
            position_ids=pos
        )[0]

        n = x.shape[1]-1
        curr = x[:, :-1].unsqueeze(2).repeat(1, 1, n, 1)  # each curr cell * n
        nxt = x[:, 1:].unsqueeze(1).repeat(1, n, 1, 1)  # n * each next cell
        # after cat: (bs, curr_idx, nxt_idx, emb_dim*2)
        pairs = torch.cat((curr, nxt), dim=3)  
        pairs = torch.cat(
            (pairs, fts[:, None, None, :].repeat(1, n, n, 1)), dim=3
        )        
        pairs = self.act(self.l0(pairs))
        pairs = self.dropout(pairs)
        pairs = self.act(self.l1(pairs))
        pairs = self.top(pairs).squeeze(-1)
        pairs.masked_fill_(label_mask.bool(), -6.5e4)
        return pairs