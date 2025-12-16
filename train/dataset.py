import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class MinecraftChunkDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer=None, context_len=24576):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                self.data.append(obj['data'])
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        self.context_len = context_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        x = self.tokenizer(chunk, return_tensors="pt", padding="max_length", truncation=True, max_length=self.context_len)
        input_ids = x["input_ids"].squeeze(0)
        attention_mask = x["attention_mask"].squeeze(0)
        # labels 通常与 input_ids 相同（自回归），可按需 mask
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }