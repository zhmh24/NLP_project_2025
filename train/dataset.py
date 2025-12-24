import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional

class MinecraftChunkDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer=None, context_length=15000, encoding_dim: Optional[int] = None):
        self.data = []
        inferred_dim = None
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                self.data.append(obj)

                if encoding_dim is None and inferred_dim is None:
                    enc = obj.get("encoding")
                    if enc is not None:
                        try:
                            inferred_dim = int(torch.as_tensor(enc).numel())
                        except Exception:
                            inferred_dim = None
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        self.context_length = context_length
        self.encoding_dim = encoding_dim if encoding_dim is not None else inferred_dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        chunk = item["data"]
        x = self.tokenizer(chunk, return_tensors="pt", padding="max_length", truncation=True, max_length=self.context_length)
        input_ids = x["input_ids"].squeeze(0)
        attention_mask = x["attention_mask"].squeeze(0)
        # labels 通常与 input_ids 相同（自回归），可按需 mask
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }

        if self.encoding_dim is not None and self.encoding_dim > 0:
            enc = item.get("encoding")
            if enc is None:
                encoding = torch.zeros(self.encoding_dim, dtype=torch.float32)
            else:
                encoding = torch.as_tensor(enc, dtype=torch.float32).view(-1)
                if encoding.numel() < self.encoding_dim:
                    encoding = torch.nn.functional.pad(encoding, (0, self.encoding_dim - encoding.numel()))
                elif encoding.numel() > self.encoding_dim:
                    encoding = encoding[: self.encoding_dim]
            batch["encoding"] = encoding

        return batch