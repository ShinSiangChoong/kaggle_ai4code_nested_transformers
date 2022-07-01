import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class MarkdownDataset(Dataset):

    def __init__(
        self,
        df_ids: pd.DataFrame,
        df_codes: pd.DataFrame,
        df_mds: pd.DataFrame,
        nb_meta: dict,
        model_name_or_path,
        max_n_codes,
        max_md_len,
        max_len
    ) -> None:
        """
        Args:
        
        """
        super().__init__()
        self.df_ids = df_ids
        self.df_codes = df_codes
        self.df_mds = df_mds
        self.nb_meta = nb_meta
        self.max_n_codes = max_n_codes
        self.max_md_len = max_md_len
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def pad_trunc(self, seq, pad_token_id=0):
        """pad and truncate the combined seq"""
        seq = seq[:self.max_len]
        if len(seq) != self.max_len:
            seq = seq + [pad_token_id] * (self.max_len - len(seq))
        return seq
        
    def process_cells(self, df, md_row, n):
        if n > self.max_n_codes:
            df = df.sample(self.max_n_codes).sort_values('code_idx').reset_index()
            n = self.max_n_codes
        if isinstance(df, pd.Series):
            df = df.to_frame().T
            
        inputs = self.tokenizer.encode_plus(
            md_row['source'],
            None,
            add_special_tokens=True,
            max_length=self.max_md_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        # decide the max_len of code cell based on the remaining space
        max_code_len = int(
            (self.max_len - len(inputs['input_ids'])) / n + 1
        )
        code_inputs = self.tokenizer.batch_encode_plus(
            df['source'].tolist(),
            add_special_tokens=True,
            max_length=max_code_len,
            padding='max_length',
            truncation=True
        )   
        
        tokens = torch.LongTensor(
            self.pad_trunc(
                inputs['input_ids'] + [
                    i for ids in code_inputs['input_ids'] for i in ids[:-1]
                ],
                pad_token_id=self.tokenizer.pad_token_id
            ),
        )
        masks = torch.LongTensor(
            self.pad_trunc(
                inputs['attention_mask'] + [
                    i for ids in code_inputs['attention_mask'] for i in ids[:-1]
                ]
            )
        )
        
        loss_ws = torch.FloatTensor(
            self.pad_trunc(
                [0] * len(inputs['input_ids']) + [
                    float(i == 1) / n
                    for ids in code_inputs['input_ids'] 
                    for i, _ in enumerate(ids[:-1])
                ]
            )
        )
        loss_ws[0] = 1
        
        labels = torch.FloatTensor(
            self.pad_trunc(
                [0] * len(inputs['input_ids']) + [
                    int(i == 1)*pct
                    for ids, pct in zip(code_inputs['input_ids'], df['rank_pct'].tolist())
                    for i, _ in enumerate(ids[:-1])
                ]
            )
        )
        labels[0] = md_row['rank_pct']
        
        return tokens, masks, labels, loss_ws

    def __getitem__(self, index):
        md = self.df_mds.loc[index]
        nb_id = md['id']
        df_codes = self.df_codes.loc[nb_id]
        tokens, masks, labels, loss_ws = self.process_cells(
            df=df_codes, 
            md_row=md, 
            n=self.nb_meta[nb_id]['n_codes'],
        )
        md_pct = torch.FloatTensor([self.nb_meta[nb_id]['md_pct']])
        
        return {
            'tokens': tokens, 
            'masks': masks, 
            'labels': labels, 
            'loss_ws': loss_ws,
            'md_pct': md_pct
        }
    
    def __len__(self) -> int:
        return self.df_mds.shape[0]
