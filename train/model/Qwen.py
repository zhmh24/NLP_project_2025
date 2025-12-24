from __future__ import annotations

from typing import Any, Dict, Optional

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

    def _ensure_encoding_proj(self, encoding: torch.Tensor) -> nn.Linear:
        encoding_dim = int(encoding.shape[-1])
        hidden_size = int(getattr(self.lm.config, "hidden_size"))
        proj = getattr(self.lm, "encoding_proj", None)
        if (
            proj is None
            or not isinstance(proj, nn.Linear)
            or proj.in_features != encoding_dim
            or proj.out_features != hidden_size
        ):
            proj = nn.Linear(encoding_dim, hidden_size, bias=False)
            proj.to(device=self.lm.device, dtype=getattr(self.lm, "dtype", torch.float32))
            self.lm.encoding_proj = proj
        return proj

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        encoding: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if encoding is None:
            return self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

        if encoding.dim() == 1:
            encoding = encoding.unsqueeze(0)

        embedding_layer = self.lm.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids)

        proj = self._ensure_encoding_proj(encoding)
        encoding = encoding.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        enc_emb = proj(encoding).unsqueeze(1)
        inputs_embeds = inputs_embeds + enc_emb

        return self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def sample(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        device: str = None,
        encoding: Optional[torch.Tensor] = None,
        **gen_kwargs,
    ):
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
            if encoding is None:
                output = self.lm.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **gen_kwargs,
                )
            else:
                input_ids = inputs["input_ids"]
                attention_mask = inputs.get("attention_mask")
                if encoding.dim() == 1:
                    encoding = encoding.unsqueeze(0)
                embedding_layer = self.lm.get_input_embeddings()
                inputs_embeds = embedding_layer(input_ids)
                proj = self._ensure_encoding_proj(encoding)
                encoding = encoding.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                inputs_embeds = inputs_embeds + proj(encoding).unsqueeze(1)
                output = self.lm.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **gen_kwargs,
                )
        generated = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated