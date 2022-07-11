import pandas as pd
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
        model_name_or_path,
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.is_train = is_train

    def __getitem__(self, index) -> dict:
        nb_id = self.df_ids.loc[index]
        if self.is_train:
            df_mds = self.df_cells.loc[nb_id, 'mark'].sample(frac=1)
            df_cell = pd.concat([self.df_cells.loc[nb_id, 'code'], df_mds])
        else:
            df_cell = self.df_cells.loc[nb_id]
        n_cells = df_cell.shape[0]
        n_pads = int(max(0, self.max_n_cells-n_cells)) + 1
        
        texts = (
            df_cell['source'].tolist() + 
            n_pads * ['padding' + self.tokenizer.sep_token]
        ) # len = max_n_cells + 1
        
        pos = torch.LongTensor(
            [0] + df_cell['pos'].tolist() + n_pads * [self.max_n_cells+2]
        )
        
        # next_cell_idx of each cell, 
        # first is a dummy start, its next_cell_idx = 1st_cell_idx
        # if it is a padded cell, next_cell_idx = max_n_cells
        labels = torch.LongTensor(
            [self.nb_meta[nb_id]['1st_cell_idx']] + 
            df_cell['next_cell_idx'].tolist() + 
            (n_pads-1) * [self.max_n_cells]
        ) # len = max_n_cells + 1
        
        # label_masks[curr, next] = mask of logit[curr, next]
        # curr ranges from [start: max_n_cells], len = max_n_cells+1 
        # next ranges from [1st_cell: last padded], len = max_n_cells+1 
        label_masks = torch.zeros(labels.shape[0], labels.shape[0], dtype=torch.bool) 
        
        # mask all padded next cells except the last
        label_masks[:, n_cells+1:] = 1
        
        # mask when next is itself
        r = torch.arange(1, n_cells+1)
        label_masks[r, r-1] = 1
        
        # Since code order is known, mask all when
        # curr is start/code and next is any code
        n_codes = self.nb_meta[nb_id]['n_codes']
        label_masks[:n_codes+1, :n_codes] = 1
        # unmask when next of code i is i+1 (only that is possible)
        label_masks[:n_codes, :n_codes][
            torch.arange(n_codes), torch.arange(n_codes)
        ] = 0
        
        # when curr is a non-last code, next won't be last pad.
        label_masks[:n_codes, n_cells] = 1
        
        # notebook padding mask
        nb_masks = torch.ones(self.max_n_cells+2)  # start + n_cells + last_pad
        nb_masks[n_cells+1:] = 0  # start to last cells are useful
        
        # start and end tokens
        start = torch.LongTensor([self.tokenizer.cls_token_id])
        
        inputs = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True
        )
        tokens = torch.LongTensor(inputs['input_ids'])
        cell_masks = torch.LongTensor(inputs['attention_mask'])
        md_pct = torch.FloatTensor([self.nb_meta[nb_id]['md_pct']])
        
        return {
            'nb_ids': nb_id,
            'tokens': tokens,
            'cell_masks': cell_masks,
            'nb_masks': nb_masks,
            'labels': labels,
            'label_masks': label_masks,
            'start': start,
            'md_pct': md_pct,
            'pos': pos
        }
    
    def __len__(self) -> int:
        return len(self.df_ids)