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
    def __init__(self, max_n_cells, d):
        super().__init__()
        self.max_n_cells = max_n_cells
        self.fc0 = Linear(d+1, 256)
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
    def __init__(self, d):
        super().__init__()
        self.fc0 = Linear(d*2, 512)
        self.fc1 = Linear(512, 128)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.LeakyReLU()
        self.top = Linear(128, 1)

    def forward(self, cells: torch.Tensor, next_masks: torch.Tensor):
        n = cells.shape[1]-1  # n = max_n_cells + 1
        curr = cells[:, :-1].unsqueeze(2).repeat(1, 1, n, 1)  # each curr cell * n
        nxt = cells[:, 1:].unsqueeze(1).repeat(1, n, 1, 1)  # n * each next cell
        # after cat: (bs, curr_idx, next_idx, dim*2)
        pairs = torch.cat((curr, nxt), dim=-1)  
        pairs = self.act(self.fc0(pairs))
        pairs = self.dropout(pairs)
        pairs = self.act(self.fc1(pairs))
        pairs = self.top(pairs).squeeze(-1)
        return pairs.masked_fill(next_masks.bool(), -6.5e4)


class CellEncoder(nn.Module):
    def __init__(self, model_path, emb_dim, n_fea):
        super(CellEncoder, self).__init__()
        self.cell_tfm = AutoModel.from_pretrained(model_path)
        self.fc0 = Linear(n_fea, emb_dim)
        self.fc1 = Linear(emb_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.act = nn.LeakyReLU()
        self.agg = Attention(emb_dim)

    def forward(self, ids, cell_masks, cell_fea):
        bs, n_cells, max_len = ids.shape
        ids = ids.view(-1, max_len)
        cell_masks = cell_masks.view(-1, max_len)
        
        # cell transformer
        tokens = self.cell_tfm(ids, cell_masks)[0]
        tokens = tokens.view(bs, n_cells, max_len, tokens.shape[-1])
        
        # cell fea
        cell_fea = self.act(self.fc0(cell_fea))
        cell_fea = self.act(self.fc1(cell_fea))
        cell_fea = self.norm(cell_fea)  # bs, n_cells, emb_dim
        # aggregate
        masks = torch.cat((
            (1-cell_masks).view(bs, n_cells, max_len, -1), 
            torch.zeros(bs, n_cells, 1, 1).bool().cuda()
        ), dim=2).bool()
        x = torch.cat((tokens, cell_fea[:, :, None, :]), dim=2)
        return self.agg(x, masks)


class NotebookModel(nn.Module):
    def __init__(self, model_path, max_n_cells, emb_dim):
        super(NotebookModel, self).__init__()
        self.cell_enc = CellEncoder(model_path, emb_dim, 2)
        self.nb_tfm = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_dim, nhead=8, batch_first=True), 
            num_layers=6
        )
        self.point_head = PointHead(max_n_cells, emb_dim)
        self.pair_head = PairHead(emb_dim)
        self.src_mask = torch.zeros(max_n_cells+2, max_n_cells+2).bool().cuda()
        for p in self.nb_tfm.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, ids, cell_masks, cell_fea, nb_atn_masks, fts, next_masks):
        cells = self.cell_enc(ids, cell_masks, cell_fea)  # bs, n_cells+2, emb_dim
        # notebook transformer
        cells = self.nb_tfm(
            cells, 
            self.src_mask,
            nb_atn_masks, 
        )#[0]
        return self.point_head(cells, fts), self.pair_head(cells, next_masks)