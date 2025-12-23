from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class MinecraftPrompter:
    def __init__(self):
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

    def get_elevation(self, attribute_data: Dict[str, Any]):
        std = attribute_data["std"]
        return std

    def get_biome(self, attribute_data: Dict[str, Any]):
        biome = attribute_data["biome"]
        return biome

    def get_log_count(self, attribute_data: Dict[str, Any]):
        log_count = attribute_data["log_count"]
        return log_count

    def elevation_prompt(self, attribute_data: Dict[str, Any]) -> str:
        std = self.get_elevation(attribute_data)
        if std >= 3.0:
            return "high elevation difference"
        else:
            return "low elevation difference"

    def biome_prompt(self, attribute_data: Dict[str, Any]) -> str:
        biome = self.get_biome(attribute_data)
        # 提取 biome 的最后一部分并将下划线替换为空格
        return biome

    def tree_prompt(self, attribute_data: Dict[str, Any]) -> str:
        log_count = self.get_log_count(attribute_data)
        if log_count >= 80:
            return "many trees"
        elif log_count > 0:
            return "little trees"
        else:
            return "no trees"
    
    def biome_transfer_id(self, biome_str: str) -> int:
        return self.biome_to_id.get(biome_str, -1)

    def tree_transfer_id(self, tree_str: str) -> int:
        return self.tree_to_id.get(tree_str, -1)
    
    def slope_transfer_id(self, slope_str: str) -> int:
        return self.slope_to_id.get(slope_str, -1)

    def generate_prompt(self, attribute_data: Dict[str, Any]) -> str:
        elevation = self.elevation_prompt(attribute_data)
        biome = self.biome_prompt(attribute_data)
        trees = self.tree_prompt(attribute_data)
        return f"Here I will provide a string, each letter represents a kind of block in Minecraft, and you need to complete a chunk. [Important]: the requirement is {elevation}, {biome} biome, {trees}. The string is: "

    # def output_hidden(self, prompt: str, device: torch.device = torch.device("cpu")):
    #     # Reducing along the first dimension to get a 768 dimensional array
    #     return (
    #         self.feature_extraction(prompt, return_tensors="pt")[0]
    #         .mean(0)
    #         .to(device)
    #         .view(1, -1)
    #     )

    def __call__(
        self, attribute_data: Dict[str, Any], device: torch.device = torch.device("cpu")
    ) -> Tuple[str, torch.Tensor]:
        prompt = self.generate_prompt(attribute_data)
        
        # hidden = self.output_hidden(prompt, device=device)
        
        return prompt