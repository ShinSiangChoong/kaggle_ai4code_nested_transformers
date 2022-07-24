import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class Linear(nn.Linear):
    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.01) 
            

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = Linear(dim, dim, bias=False)
        self.v = Linear(dim, 1, bias=False)

    def forward(self, keys: torch.Tensor, masks: torch.Tensor):
        weights = self.v(torch.tanh(self.W(keys)))
        weights.masked_fill_(masks, -6.5e4)
        weights = F.softmax(weights, dim=2)
        return torch.sum(weights * keys, dim=2)


class PointHead(nn.Module):
    def __init__(self, max_n_cells, emb_dim):
        super().__init__()
        self.max_n_cells = max_n_cells
        self.fc0 = Linear(emb_dim+1, 256)
        self.fc1 = Linear(256, 128)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.LeakyReLU()
        self.top = Linear(128, 1)    

    def forward(self, cells: torch.Tensor, fts: torch.Tensor):
        x = torch.cat((cells[:, 1:-1], fts.unsqueeze(1).repeat(1, self.max_n_cells, 1)), 2)
        x = self.act(self.fc0(x))
        x = self.dropout(x)
        x = self.act(self.fc1(x))
        return self.top(x).squeeze(-1)
        

class PairHead(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.fc0 = Linear(emb_dim*2, 512)
        self.fc1 = Linear(512, 128)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.LeakyReLU()
        self.top = Linear(128, 1)

    def forward(self, cells: torch.Tensor, next_masks: torch.Tensor):
        n = cells.shape[1]-1  # n = max_n_cells + 1
        curr = cells[:, :-1].unsqueeze(2).repeat(1, 1, n, 1)  # each curr cell * n
        nxt = cells[:, 1:].unsqueeze(1).repeat(1, n, 1, 1)  # n * each next cell
        # after cat: (bs, curr_idx, next_idx, emb_dim*2)
        pairs = torch.cat((curr, nxt), dim=-1)  
        pairs = self.act(self.fc0(pairs))
        pairs = self.dropout(pairs)
        pairs = self.act(self.fc1(pairs))
        pairs = self.top(pairs).squeeze(-1)
        return pairs.masked_fill(next_masks.bool(), -6.5e4)


class NotebookModel(nn.Module):
    def __init__(self, model_path, max_n_cells, emb_dim):
        super(NotebookModel, self).__init__()
        self.cell_tfm = AutoModel.from_pretrained(model_path)
        self.nb_tfm = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_dim, nhead=8, batch_first=True), 
            num_layers=6
        )
        self.agg = Attention(emb_dim)
        self.point_head = PointHead(max_n_cells, emb_dim)
        self.pair_head = PairHead(emb_dim)
        self.src_mask = torch.zeros(max_n_cells+2, max_n_cells+2).bool().cuda()
        for p in self.nb_tfm.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_cells(self, ids, cell_masks):
        bs, n_cells, max_len = ids.shape
        ids = ids.view(-1, max_len)
        cell_masks = cell_masks.view(-1, max_len)
        
        # cell transformer
        x = self.cell_tfm(ids, cell_masks)[0]
        x = x.view(bs, n_cells, max_len, x.shape[-1])
        return self.agg(x, (1-cell_masks).bool().view(bs, n_cells, max_len, -1))

    def forward(self, ids, cell_masks, nb_atn_masks, fts, next_masks):
        cells = self.encode_cells(ids, cell_masks)  # n_cells + 2
        # notebook transformer
        cells = self.nb_tfm(
            cells, 
            self.src_mask,
            nb_atn_masks, 
        )#[0]
        return self.point_head(cells, fts), self.pair_head(cells, next_masks)