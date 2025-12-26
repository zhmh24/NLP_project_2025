import os
from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, PreTrainedModel
from torch.amp import GradScaler, autocast  # Updated import

from train.model.Qwen import QwenForMinecraft
from train.model.base import BaseMinecraftLM
from train.dataset import MinecraftChunkDataset
from train.prompter import MinecraftPrompter

@dataclass
class TrainingConfig:
    output_dir: str = "Qwen-MinecraftLM"
    data_dir: str = "generated_data"
    learning_rate: float = 5e-5
    epsilon: float = 1e-8
    lr_warmup_steps: int = 50
    batch_size: int = 4
    total_steps: int = 1000
    eval_iteration: int = 100
    save_iteration: int = 2500
    gradient_accumulation_steps: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def pretty_print(self):
        print("================== Training Config ==================")
        d = asdict(self)
        for k in d:
            print(f"{k} -- {d[k]}")
        print("================== MinecraftLM ==================")

class MinecraftTrainer:
    def __init__(self, model: BaseMinecraftLM, dataset: MinecraftChunkDataset, test_dataset: MinecraftChunkDataset, config: Optional[TrainingConfig] = None):
        self.model = model
        self.prompter = MinecraftPrompter()
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.config = config or TrainingConfig()
        self.device = self.config.device
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate, eps=self.config.epsilon)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=self.config.total_steps,
        )
        self.scaler = GradScaler('cuda')  # Updated GradScaler initialization

    def evaluate(self, dataset):
        """
        Evaluate the model on a given dataset.

        Args:
            dataset (MinecraftChunkDataset): The dataset to evaluate on.

        Returns:
            float: The average loss on the dataset.
        """
        self.model.eval()  # 切换到评估模式
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        total_loss = 0
        num_batches = 0

        with torch.no_grad():  # 禁用梯度计算
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                biome_ids = batch["biome_ids"].to(self.device)
                tree_ids = batch["tree_ids"].to(self.device)
                elevation_ids = batch["elevation_ids"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    biome_ids=biome_ids,
                    elevation_ids=elevation_ids,
                    tree_ids=tree_ids
                )
                loss = outputs.loss
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Evaluation Loss: {avg_loss:.4f}")
        return avg_loss

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
                biome_ids = batch["biome_ids"].to(self.device)
                tree_ids = batch["tree_ids"].to(self.device)
                elevation_ids = batch["elevation_ids"].to(self.device)

                with autocast('cuda'):  # 启用混合精度训练
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        biome_ids=biome_ids,
                        elevation_ids=elevation_ids,
                        tree_ids=tree_ids
                    )
                    loss = outputs.loss

                self.scaler.scale(loss).backward()  
                self.scaler.step(self.optimizer)  
                self.lr_scheduler.step()  # Scheduler step after optimizer step
                self.scaler.update()  # 更新缩放因子
                self.optimizer.zero_grad()

                bar.set_postfix({"loss": loss.item(), "lr": self.lr_scheduler.get_last_lr()[0]})
                step += 1

                if step % self.config.eval_iteration == 0:
                    print(f"Step {step}: loss={loss.item():.4f}")
                    print(f"\n--- Generating Sample at Step {step} ---")
                    try:
                        
                        last_idx = batch['idx'][-1].item()
                        
                        full_training_text = self.dataset.data[last_idx] 
                        attr_data = self.dataset.get_attribute_data(last_idx)
                        
                        
                        full_text = full_training_text
                        
                        
                        first_dollar = full_text.find('$')
                        second_dollar = full_text.find('$', first_dollar + 1)
                        
                        if second_dollar != -1:
                            input_prompt = full_text[:second_dollar + 1] 
                        else:
                            input_prompt = full_text[:first_dollar + 1]

                        print(f"Captured Prompt:\n{input_prompt}")

                        raw_biome = attr_data['biome'] 
                        raw_tree = self.prompter.tree_prompt(attr_data)
                        raw_slope = self.prompter.elevation_prompt(attr_data)
                        print(f"The attributes are:\n{raw_biome}, {raw_tree}, {raw_slope}\n")

                        generated_body = self.model.sample_stepwise(
                            prompt=input_prompt,
                            biome_str=raw_biome,
                            tree_str=raw_tree,
                            slope_str=raw_slope,
                            max_new_tokens=2000, 
                            temperature=0.8
                        )



                        save_filename = os.path.join(self.config.data_dir, f"sample_step_{step}.txt")
                        with open(save_filename, "w", encoding="utf-8") as f:
                            f.write(f"--- Metadata ---\n")
                            f.write(f"Step: {step}\n")
                            f.write(f"Attributes: {attr_data}\n")
                            f.write(f"--- Prompt ---\n")
                            f.write(input_prompt + "\n")
                            f.write(f"--- Generated ---\n")
                            f.write(generated_body + "\n")
                        print(f"✅ Sample saved to {save_filename}")

                    except Exception as e:
                        print(f"Sample generation failed: {e}")
                    finally:
                        self.model.train() 

                if step % self.config.save_iteration == 0:
                    self.model.save_model(self.config.output_dir, step)
                    print(f"Model saved at step {step}")

                if step >= self.config.total_steps:
                    print("Training finished.")
                    return
