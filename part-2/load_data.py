import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt', quiet=True)
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
TASK_PREFIX = "translate English to SQL: "
# Longer contexts help when NL/SQL are truncated (hurts F1 if cut mid-query).
MAX_NL_LEN = 256
MAX_SQL_LEN = 384


class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f"{split}.nl")
        nl_lines = load_lines(nl_path)

        examples = []
        pad_id = tokenizer.pad_token_id

        if split == "test":
            for nl in nl_lines:
                enc = tokenizer(
                    TASK_PREFIX + nl,
                    truncation=True,
                    max_length=MAX_NL_LEN,
                    return_tensors="pt",
                )
                examples.append(
                    {
                        "input_ids": enc["input_ids"].squeeze(0),
                        "attention_mask": enc["attention_mask"].squeeze(0),
                    }
                )
            return examples

        sql_path = os.path.join(data_folder, f"{split}.sql")
        sql_lines = load_lines(sql_path)
        assert len(nl_lines) == len(sql_lines), "NL and SQL line counts must match"

        eos_id = tokenizer.eos_token_id

        for nl, sql in zip(nl_lines, sql_lines):
            enc = tokenizer(
                TASK_PREFIX + nl,
                truncation=True,
                max_length=MAX_NL_LEN,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)

            sql_enc = tokenizer(
                sql,
                truncation=True,
                max_length=MAX_SQL_LEN,
                add_special_tokens=False,
            )
            sql_ids = list(sql_enc["input_ids"])
            if len(sql_ids) == 0 or sql_ids[-1] != eos_id:
                sql_ids.append(eos_id)

            sql_token_ids = sql_ids
            decoder_inputs = [pad_id] + sql_token_ids[:-1]
            decoder_targets = sql_token_ids

            examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "decoder_inputs": torch.tensor(decoder_inputs, dtype=torch.long),
                    "decoder_targets": torch.tensor(decoder_targets, dtype=torch.long),
                }
            )
        return examples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def normal_collate_fn(batch):
    encoder_ids = pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=PAD_IDX
    )
    encoder_mask = pad_sequence(
        [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
    )
    decoder_inputs = pad_sequence(
        [b["decoder_inputs"] for b in batch], batch_first=True, padding_value=PAD_IDX
    )
    decoder_targets = pad_sequence(
        [b["decoder_targets"] for b in batch], batch_first=True, padding_value=PAD_IDX
    )
    initial_decoder_inputs = decoder_inputs[:, 0:1]
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    encoder_ids = pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=PAD_IDX
    )
    encoder_mask = pad_sequence(
        [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
    )
    initial_decoder_inputs = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)
    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    raise NotImplementedError("load_prompting_data is not implemented in this assignment skeleton.")
