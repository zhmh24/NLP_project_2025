from __future__ import annotations

from typing import Any, Dict, List, Optional

from typing import Any, Dict, List, Optional

import torch
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

    def sample(self, prompt: str, max_new_tokens: int = 512, temperature: float = 1.0, top_p: float = 0.95, device: str = None, **gen_kwargs):
        """
        基于prompt生成区块文本。
        :param prompt: 输入的文本提示
        :param max_new_tokens: 生成的最大token数
        :param temperature: 采样温度
        :param top_p: nucleus采样参数
        :param device: 指定推理设备
        :param gen_kwargs: 其他transformers generate参数
        :return: 生成的字符串
        """
        device = device or (self.lm.device if hasattr(self.lm, 'device') else 'cpu')
        self.lm.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = self.lm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **gen_kwargs
            )
        generated = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated