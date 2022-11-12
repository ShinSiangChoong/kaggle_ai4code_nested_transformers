import pandas as pd
import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class NotebookDataset(Dataset):
    def __init__(
        self,
        df_ids: pd.DataFrame,
        df_cells: pd.DataFrame,
        nb_meta: dict,
        tokenizer_name_or_path,
        max_n_cells,
        max_len,
        ellipses_token_id,
        is_train
    ) -> None:
        """
        Args:
        
        """
        super().__init__()
        self.df_ids = df_ids
        self.df_cells = df_cells
        self.nb_meta = nb_meta
        self.max_n_cells = max_n_cells
        self.max_len = max_len
        self.front_lim = (max_len-2) // 2 + 2 - (max_len%2 == 0)
        self.back_lim = self.max_len - self.front_lim - 1
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.ellipses_token_id = ellipses_token_id
        self.is_train = is_train

    def determine_next_cell_idx(self, df_cell, n_cells, n_pads):
        """
        Determine next_cell_idx for each cell (based on the correct cell order)
        Shuffled cells: <start> <cell_a> <cell_b> ... <pad> <pad> ... <end>
        next_cell_idx:  <cell_1> <a.next> <b.next> ... <max_n_cells> <max_n_cells> ...
        next_cell_idx of <start> = the 1st cell (based on correct cell order)
        next_cell_idx of <pad> = max_n_cells (ignored in loss calculation)
        """ 

        df_cell['cell_idx'] = np.arange(n_cells)  # [0, n_cells-1]
        df_cell['cell_idx'] = df_cell['cell_idx'].astype(int)
        df_tmp = df_cell[['cell_idx', 'rank']].sort_values('rank').reset_index(drop=True)
        df_tmp['next_cell_idx'] = df_tmp['cell_idx'].shift(-1).fillna(n_cells).astype(int)
        df_cell['next_cell_idx'] = df_cell['cell_idx'].map(
            df_tmp.set_index('cell_idx')['next_cell_idx']
        )

        return torch.LongTensor(
            [df_tmp['cell_idx'].iloc[0]] + 
            df_cell['next_cell_idx'].tolist() + 
            n_pads * [self.max_n_cells]
        )  # len = max_n_cells + 1

    def create_pairwise_mask(self, max_n_cells, n_cells, nb_id):
        """
        Create pairwise boolean mask (1 = impossible, 0 = possible) 
        Since the code cell order is known, certain cells can't be the next cell of certain cells
        """
        # label_masks[curr, next] = mask of logit[curr, next]
        # curr ranges from [start: max_n_cells], len = max_n_cells+1 
        # next ranges from [1st_cell: end], len = max_n_cells+1 
        next_masks = torch.zeros(
            max_n_cells+1, max_n_cells+1, dtype=torch.bool
        )    

        # when curr is padded, ignore all non-max_n_cells next
        next_masks[1+n_cells:, :max_n_cells] = 1        
        # mask all padded next cells
        next_masks[:, 1+n_cells:] = 1
        
        # mask when next is itself
        r = torch.arange(1, n_cells+1)
        next_masks[r, r-1] = 1
        
        # mask all when curr is start/code and next is any code
        n_codes = self.nb_meta[nb_id]['n_codes']
        next_masks[:n_codes+1, :n_codes] = 1
        # unmask only when next of code i is i+1 (only that is possible)
        next_masks[:n_codes, :n_codes][
            torch.arange(n_codes), torch.arange(n_codes)
        ] = 0

        # when curr is a non-last code, next won't be end.
        next_masks[:n_codes, n_cells] = 1
        return next_masks

    def trunc_mid(self, ids):
        """
        Truncate the middle part of the texts if it is too long
        Use a token (ellipses_token_id) to separate the front and back part
        """
        if len(ids) > self.max_len:
            return ids[:self.front_lim] + [int(self.ellipses_token_id)] + ids[-self.back_lim:]
        return ids

    def encode_texts(self, df_cell, n_pads):
        texts = (
            ['starting' + self.tokenizer.sep_token] +
            df_cell['source'].tolist() + 
            ['ending' + self.tokenizer.sep_token] +
            n_pads * ['padding' + self.tokenizer.sep_token]
        )  # len = max_n_cells + 2

        inputs = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=False
        )
        tokens = list(map(self.trunc_mid, inputs['input_ids']))
        tokens = torch.LongTensor(tokens)
        cell_masks = list(map(lambda x: x[:self.max_len], inputs['attention_mask']))
        cell_masks = torch.LongTensor(cell_masks)
        return tokens, cell_masks

    def create_cell_features(self, df_cell, n_pads):
        """
        Construct the two cell features:
        1. is_codes (boolean)
        2. rel_pos [0:1]
        """
        is_codes = [1] + df_cell['is_code'].tolist() + (n_pads+1)*[0]  
        rel_pos = [0] + df_cell['rel_pos'].tolist() + (n_pads+1)*[0]  # len = max_n_cells+2
        return torch.stack(
            (torch.FloatTensor(is_codes), torch.FloatTensor(rel_pos)), dim=-1
        )

    def __getitem__(self, index) -> dict:
        nb_id = self.df_ids.loc[index]
        n_cells = self.nb_meta[nb_id]['n_cells']
        if self.is_train:
            df_cell = self.df_cells.loc[nb_id].copy()
            n_pads = int(max(0, self.max_n_cells-n_cells))
            max_n_cells = self.max_n_cells
        else:
            df_cell = self.df_cells.loc[nb_id].copy()
            max_n_cells = n_cells
            n_pads = 0

        tokens, cell_masks = self.encode_texts(df_cell, n_pads)
        cell_fea = self.create_cell_features(df_cell, n_pads)

        next_indices = self.determine_next_cell_idx(df_cell, n_cells, n_pads)
        next_masks = self.create_pairwise_mask(max_n_cells, n_cells, nb_id)

        # notebook padding masks
        nb_atn_masks = torch.zeros(max_n_cells+2).bool()  # start + n_cells + end
        nb_atn_masks[n_cells+2:] = True  # start to end are useful
        nb_cls_masks = torch.ones(max_n_cells+1).bool()
        nb_cls_masks[n_cells+1:] = False
        nb_reg_masks = torch.ones(max_n_cells).bool()
        nb_reg_masks[n_cells:] = False

        md2code_masks = torch.ones(
            next_indices.shape[0], next_indices.shape[0], dtype=torch.bool
        )
        md2code_masks[:, :self.nb_meta[nb_id]['n_codes']] = 0

        md_pct = torch.FloatTensor([self.nb_meta[nb_id]['md_pct']])
        n_mds = torch.FloatTensor([self.nb_meta[nb_id]['n_mds']])
        pct_ranks = torch.FloatTensor(df_cell['pct_rank'].tolist() + n_pads*[0])
        
        return {
            'nb_ids': nb_id,
            'tokens': tokens,
            'cell_masks': cell_masks,
            'nb_atn_masks': nb_atn_masks,
            'nb_cls_masks': nb_cls_masks,
            'nb_reg_masks': nb_reg_masks,
            'cell_fea': cell_fea,
            'next_indices': next_indices,
            'next_masks': next_masks,
            'md2code_masks': md2code_masks,
            'pct_ranks': pct_ranks,
            'md_pct': md_pct,
            'n_mds': n_mds
        }
    
    def __len__(self) -> int:
        return len(self.df_ids)