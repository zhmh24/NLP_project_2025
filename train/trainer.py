import os
from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, PreTrainedModel

from train.model.Qwen import QwenForMinecraft
from train.model.base import BaseMinecraftLM
from train.dataset import MinecraftChunkDataset

@dataclass
class TrainingConfig:
    output_dir: str = "Qwen-MinecraftLM"
    learning_rate: float = 5e-5
    epsilon: float = 1e-8
    lr_warmup_steps: int = 50
    batch_size: int = 4
    total_steps: int = 1000
    eval_iteration: int = 100
    save_iteration: int = 1000
    gradient_accumulation_steps: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def pretty_print(self):
        print("================== Training Config ==================")
        d = asdict(self)
        for k in d:
            print(f"{k} -- {d[k]}")
        print("================== MinecraftLM ==================")

class MinecraftTrainer:
    def __init__(self, model: BaseMinecraftLM, dataset: MinecraftChunkDataset, config: Optional[TrainingConfig] = None):
        self.model = model
        self.dataset = dataset
        self.config = config or TrainingConfig()
        self.device = self.config.device
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.lm.parameters(), lr=self.config.learning_rate, eps=self.config.epsilon)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=self.config.total_steps,
        )

    def train(self):
        self.model.train()
        dataloader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        step = 0
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.config.pretty_print()
        for epoch in range(9999):  # 无限epoch直到step满足
            bar = tqdm(dataloader, desc=f"Epoch {epoch}")
            for batch in bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model.lm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                bar.set_postfix({"loss": loss.item(), "lr": self.lr_scheduler.get_last_lr()[0]})
                step += 1

                if step % self.config.eval_iteration == 0:
                    print(f"Step {step}: loss={loss.item():.4f}")

                if step % self.config.save_iteration == 0:
                    self.model.save_model(self.config.output_dir, step)
                    print(f"Model saved at step {step}")

                if step >= self.config.total_steps:
                    print("Training finished.")
                    return
