import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from train.prompter import MinecraftPrompter

class MinecraftChunkDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer=None, context_length=15000):
        self.data = []
        self.std_data = []
        self.biome_data = []
        self.log_data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                self.data.append(obj['data'])
                self.std_data.append(obj['std'])
                self.biome_data.append(obj['biome'])
                self.log_data.append(obj['log_count'])
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        self.context_length = context_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        attr = self.get_attribute_data(idx)

        # Generate text prompt for llm
        prompter = MinecraftPrompter()
        prompt = prompter.generate_prompt(attr)
        chunk = prompt + chunk

        x = self.tokenizer(chunk, return_tensors="pt", padding="max_length", truncation=True, max_length=self.context_length)
        input_ids = x["input_ids"].squeeze(0)
        attention_mask = x["attention_mask"].squeeze(0)
        # labels 通常与 input_ids 相同（自回归），可按需 mask
        biome_id = prompter.biome_transfer_id(prompter.biome_prompt(attr))
        tree_id = prompter.tree_transfer_id(prompter.tree_prompt(attr))
        slope_id = prompter.slope_transfer_id(prompter.elevation_prompt(attr))
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
            "biome_ids": biome_id,
            "elevation_ids": slope_id,
            "tree_ids": tree_id,
            "idx": idx
        }

    def get_attribute_data(self, idx):
        return {
            "std": self.std_data[idx],
            "biome": self.biome_data[idx],
            "log_count": self.log_data[idx]
        }