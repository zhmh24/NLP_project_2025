from __future__ import annotations

from typing import Any, Dict, List, Optional

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from train.model.base import BaseMinecraftLM

PRETRAINED_MODEL_PATH = "Qwen/Qwen3-0.6B"

class QwenForMinecraft(BaseMinecraftLM, nn.Module):
    from tqdm import tqdm

    PRETRAINED_LM_PATH = PRETRAINED_MODEL_PATH
    PRETRAINED_TOKENIZER_PATH = PRETRAINED_MODEL_PATH

    BASE_LM_PATH = PRETRAINED_MODEL_PATH
    BASE_TOKENIZER_PATH = PRETRAINED_MODEL_PATH

    def __init__(
        self,
        lm: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        context_length: int = 24576,
        lm_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        lm_kwargs: Dict[str, Any] = {},
        tokenizer_kwargs: Dict[str, Any] = {},
    ):
        nn.Module.__init__(self)

        super().__init__(
            lm,
            tokenizer,
            context_length,
            lm_path,
            tokenizer_path,
            lm_kwargs,
            tokenizer_kwargs,
        )

        h_size = 1024

        self.biome_embedding = nn.Embedding(100, h_size)
        self.elevation_embedding = nn.Embedding(10, h_size)
        self.tree_embedding = nn.Embedding(10, h_size)


        self.biome_to_id = {
  "universal_minecraft:plains": 0,
  "universal_minecraft:savanna": 1,
  "universal_minecraft:flower_forest": 2,
  "universal_minecraft:giant_tree_taiga": 3,
  "universal_minecraft:sunflower_plains": 4,
  "universal_minecraft:jungle": 5,
  "universal_minecraft:birch_forest": 6,
  "universal_minecraft:desert": 7,
  "universal_minecraft:eroded_badlands": 8,
  "universal_minecraft:forest": 9,
  "universal_minecraft:snowy_beach": 10,
  "universal_minecraft:meadow": 11,
  "universal_minecraft:snowy_taiga": 12,
  "universal_minecraft:snowy_slopes": 13,
  "universal_minecraft:taiga": 14,
  "universal_minecraft:giant_spruce_taiga": 15,
  "universal_minecraft:snowy_tundra": 16,
  "universal_minecraft:ocean": 17,
  "universal_minecraft:beach": 18,
  "universal_minecraft:swamp": 19,
  "universal_minecraft:wooded_badlands_plateau": 20,
  "universal_minecraft:tall_birch_forest": 21,
  "universal_minecraft:badlands": 22,
  "universal_minecraft:frozen_peaks": 23,
  "universal_minecraft:bamboo_jungle": 24,
  "universal_minecraft:mangrove_swamp": 25,
  "universal_minecraft:frozen_river": 26,
  "universal_minecraft:jungle_edge": 27,
  "universal_minecraft:river": 28,
  "universal_minecraft:jagged_peaks": 29,
  "universal_minecraft:cherry_grove": 30,
  "universal_minecraft:frozen_ocean": 31,
  "universal_minecraft:stony_peaks": 32,
  "universal_minecraft:wooded_mountains": 33,
  "universal_minecraft:savanna_plateau": 34,
  "universal_minecraft:dark_forest": 35,
  "universal_minecraft:grove": 36,
  "universal_minecraft:stone_shore": 37,
  "universal_minecraft:deep_frozen_ocean": 38,
  "universal_minecraft:shattered_savanna": 39,
  "universal_minecraft:mountains": 40,
  "universal_minecraft:ice_spikes": 41,
  "universal_minecraft:dripstone_caves": 42,
  "universal_minecraft:cold_ocean": 43,
  "universal_minecraft:mushroom_fields": 44,
  "universal_minecraft:gravelly_mountains": 45,
  "universal_minecraft:lukewarm_ocean": 46,
  "universal_minecraft:warm_ocean": 47,
  "universal_minecraft:lush_caves": 48
}
        self.tree_to_id = {"no trees": 0, "little trees": 1, "many trees": 2}
        self.slope_to_id = {"low elevation difference": 0, "high elevation difference": 1}
    
    def forward(self, input_ids, attention_mask, labels, biome_ids=None, elevation_ids=None, tree_ids=None):
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        biome_embeds = self.biome_embedding(biome_ids).unsqueeze(1).repeat(1, seq_length, 1) if biome_ids is not None else torch.zeros((batch_size, seq_length, self.lm.config.hidden_size), device=device)
        elevation_embeds = self.elevation_embedding(elevation_ids).unsqueeze(1).repeat(1, seq_length, 1) if elevation_ids is not None else torch.zeros((batch_size, seq_length, self.lm.config.hidden_size), device=device)
        tree_embeds = self.tree_embedding(tree_ids).unsqueeze(1).repeat(1, seq_length, 1) if tree_ids is not None else torch.zeros((batch_size, seq_length, self.lm.config.hidden_size), device=device)

        inputs_embeds = self.lm.get_input_embeddings()(input_ids)
        inputs_embeds = inputs_embeds + biome_embeds + elevation_embeds + tree_embeds

        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

    def generate_seed(self, length: int, batch_size: Optional[int] = None):
        seed = self.tokenizer("X", return_tensors="pt").input_ids.squeeze()
        if batch_size is None:
            return seed.repeat(length)
        return seed.view(1, 1).repeat(batch_size, length)

    def load_pretrained_lm(self, path: str, lm_kwargs: Dict[str, Any]) -> PreTrainedModel:
        return AutoModelForCausalLM.from_pretrained(path, **lm_kwargs)

    def load_pretrained_tokenizer(
        self, path: str, tokenizer_kwargs: Dict[str, Any]
    ) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained(path, **tokenizer_kwargs)

    def sample(self, prompt: str, biome_str: str, tree_str: str, slope_str: str, 
               max_new_tokens: int = 9216, device: str = None, **gen_kwargs):
        """
        采样时每步都加条件 embedding，和训练完全一致。
        """
        device = device or (self.lm.device if hasattr(self.lm, 'device') else 'cpu')
        self.lm.eval()

        # 1. 编码文本
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # 2. 条件 id
        biome_id = self.biome_to_id.get(biome_str, -1)
        tree_id = self.tree_to_id.get(tree_str, -1)
        slope_id = self.slope_to_id.get(slope_str, -1)

        # 3. 保存原始 forward
        original_forward = self.lm.forward

        def custom_forward(input_ids=None, inputs_embeds=None, **kwargs):
            # input_ids: (batch, seq)
            if inputs_embeds is None and input_ids is not None:
                seq_len = input_ids.shape[1]
                b_id = torch.full((input_ids.shape[0], seq_len), biome_id, dtype=torch.long, device=input_ids.device)
                t_id = torch.full((input_ids.shape[0], seq_len), tree_id, dtype=torch.long, device=input_ids.device)
                s_id = torch.full((input_ids.shape[0], seq_len), slope_id, dtype=torch.long, device=input_ids.device)
                embeds = self.lm.get_input_embeddings()(input_ids)
                biome_embeds = self.biome_embedding(b_id)
                tree_embeds = self.tree_embedding(t_id)
                elevation_embeds = self.elevation_embedding(s_id)
                inputs_embeds = embeds + biome_embeds + elevation_embeds + tree_embeds
            return original_forward(input_ids=None, inputs_embeds=inputs_embeds, **kwargs)

        self.lm.forward = custom_forward
        try:
            with torch.no_grad():
                output = self.lm.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **gen_kwargs
                )
        finally:
            self.lm.forward = original_forward
        return self.tokenizer.decode(output[0], skip_special_tokens=True)



    def step_sample(
        self,
        input_ids: torch.Tensor,
        biome_id: int,
        tree_id: int,
        slope_id: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        use_argmax: bool = False,
    ):
        # input_ids: (batch, seq)
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        with torch.amp.autocast('cuda'):
            # 构造条件 embedding
            b_id = torch.full((batch_size, seq_len), biome_id, dtype=torch.long, device=device)
            t_id = torch.full((batch_size, seq_len), tree_id, dtype=torch.long, device=device)
            s_id = torch.full((batch_size, seq_len), slope_id, dtype=torch.long, device=device)
            embeds = self.lm.get_input_embeddings()(input_ids)
            biome_embeds = self.biome_embedding(b_id)
            tree_embeds = self.tree_embedding(t_id)
            elevation_embeds = self.elevation_embedding(s_id)
            inputs_embeds = embeds + biome_embeds + elevation_embeds + tree_embeds

            attention_mask = torch.ones_like(input_ids, device=device)
            outputs = self.lm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            logits = outputs.logits[:, -1, :] / temperature

            # Top-k, top-p采样
            if use_argmax:
                next_token = logits.argmax(-1)
            else:
                # top_k
                if top_k > 0:
                    top_k = min(top_k, logits.size(-1))
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits = logits.masked_fill(indices_to_remove, float('-inf'))
                # top_p
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    for batch_idx in range(logits.size(0)):
                        remove_indices = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                        logits[batch_idx, remove_indices] = float('-inf')
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        return next_token

    def sample_stepwise(
        self,
        prompt: str,
        biome_str: str,
        tree_str: str,
        slope_str: str,
        max_new_tokens: int = 2000,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        use_argmax: bool = False,
        device: str = None,
        show_progress: bool = True,
    ):
        from tqdm import tqdm
        device = device or (self.lm.device if hasattr(self.lm, 'device') else 'cpu')
        self.lm.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        biome_id = self.biome_to_id.get(biome_str, -1)
        tree_id = self.tree_to_id.get(tree_str, -1)
        slope_id = self.slope_to_id.get(slope_str, -1)

        generated = input_ids
        iterator = tqdm(range(max_new_tokens), desc="Sampling") if show_progress else range(max_new_tokens)
        for _ in iterator:
            next_token = self.step_sample(
                generated,
                biome_id,
                tree_id,
                slope_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                use_argmax=use_argmax,
            )
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
            # 可选：遇到eos提前结束
            if (next_token == self.tokenizer.eos_token_id).all():
                break
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)