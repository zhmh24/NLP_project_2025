from __future__ import annotations

from typing import Any, Dict, List, Optional

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from train.model.base import BaseMinecraftLM

PRETRAINED_MODEL_PATH = "Qwen/Qwen3-0.6B"

class QwenForMinecraft(BaseMinecraftLM):

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
        device = device or self.lm.device
        self.lm.eval()
        
        # 1. 编码文本
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        # input_ids = inputs["input_ids"]
        
        # 2. 准备条件 ID
        b_id = torch.tensor([self.biome_to_id.get(biome_str, 0)]).to(device)
        t_id = torch.tensor([self.tree_to_id.get(tree_str, 0)]).to(device)
        s_id = torch.tensor([self.slope_to_id.get(slope_str, 0)]).to(device)
        
        # 3. 构造 inputs_embeds
        # 注意：generate 过程中需要处理不断增长的序列，这里使用最简化的 inputs_embeds 传入方式

        cond_vec = (self.biome_embedding(b_id) + self.tree_embedding(t_id) + self.elevation_embedding(s_id)).unsqueeze(1)

        original_forward = self.lm.forward

        def hooked_forward(input_ids=None, inputs_embeds=None, **kwargs):
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds = self.lm.get_input_embeddings()(input_ids)
                inputs_embeds = inputs_embeds + cond_vec
            return original_forward(input_ids=None, inputs_embeds=inputs_embeds, **kwargs)
        
        self.lm.forward = hooked_forward
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        try:
            with torch.no_grad():
                output = self.lm.generate(
                    input_ids=inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    **gen_kwargs
                )
        finally:
            self.lm.forward = original_forward
        return self.tokenizer.decode(output[0], skip_special_tokens=True)