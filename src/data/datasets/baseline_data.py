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
        self.is_train = is_train

    def trunc_mid(self, ids):
        if len(ids) > self.max_len:
            return ids[:self.front_lim] + [1873] + ids[-self.back_lim:]
        return ids

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
            # mod = (n_cells+2) % 8
            # n_pads = int((mod != 0)*(8 - mod))
            # max_n_cells = n_cells + n_pads

        
        texts = (
            ['starting' + self.tokenizer.sep_token] +
            df_cell['source'].tolist() + 
            ['ending' + self.tokenizer.sep_token] +
            n_pads * ['padding' + self.tokenizer.sep_token]
        )  # len = max_n_cells + 2
        
        # pos = torch.LongTensor(
        #     [0] + df_cell['pos'].tolist() + [self.max_n_cells+1] 
        #     + n_pads*[self.max_n_cells+2]
        # )
        
        # next_cell_idx of each cell, 
        # first is a dummy start, its next_cell_idx = 1st_cell_idx
        # if it is a padded cell, next_cell_idx = self.max_n_cells (ignore anyway)
        df_cell['cell_idx'] = np.arange(n_cells)  # [0, n_cells-1]
        df_tmp = df_cell[['cell_idx', 'rank']].sort_values('rank')
        df_cell['cell_idx'] = df_cell['cell_idx'].astype(int)
        df_tmp['next_cell_idx'] = df_tmp['cell_idx'].shift(-1).fillna(n_cells).astype(int)
        df_cell = df_cell.merge(
            df_tmp.drop('rank', axis=1), on='cell_idx', how='left'
        )
        next_indices = torch.LongTensor(
            [df_tmp['cell_idx'].iloc[0]] + 
            df_cell['next_cell_idx'].tolist() + 
            n_pads * [self.max_n_cells]
        )  # len = max_n_cells + 1

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
        
        # Since code order is known, mask all when
        # curr is start/code and next is any code
        n_codes = self.nb_meta[nb_id]['n_codes']
        next_masks[:n_codes+1, :n_codes] = 1
        # unmask when next of code i is i+1 (only that is possible)
        next_masks[:n_codes, :n_codes][
            torch.arange(n_codes), torch.arange(n_codes)
        ] = 0

        # when curr is a non-last code, next won't be end.
        next_masks[:n_codes, n_cells] = 1

        md2code_masks = torch.ones(
            next_indices.shape[0], next_indices.shape[0], dtype=torch.bool
        )
        md2code_masks[:, :n_codes] = 0

        # notebook padding mask
        nb_atn_masks = torch.zeros(max_n_cells+2).bool()  # start + n_cells + end
        nb_atn_masks[n_cells+2:] = True  # start to end are useful
        nb_cls_masks = torch.ones(max_n_cells+1).bool()
        nb_cls_masks[n_cells+1:] = False
        nb_reg_masks = torch.ones(max_n_cells).bool()
        nb_reg_masks[n_cells:] = False
        
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

        # cell features, len = max_n_cells + 2
        is_codes = [1] + df_cell['is_code'].tolist() + (n_pads+1)*[0]  
        rel_pos = [0] + df_cell['rel_pos'].tolist() + (n_pads+1)*[0]  # len = max_n_cells+2
        cell_fea = torch.stack(
            (torch.FloatTensor(is_codes), torch.FloatTensor(rel_pos)), dim=-1
        )

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
            # 'pos': pos
        }
    
    def __len__(self) -> int:
        return len(self.df_ids)