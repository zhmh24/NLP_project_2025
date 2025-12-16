from typing import Optional, Union

from transformers import PreTrainedModel, PreTrainedTokenizer

from train.model.base import BaseMinecraftLM
from train.model.Qwen import QwenForMinecraft

def MinecraftLM(
    lm: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    context_length: int = 24576,
    lm_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    **kwargs,
):
    return QwenForMinecraft(
        lm=lm,
        tokenizer=tokenizer,
        context_length=context_length,
        lm_path=lm_path,
        tokenizer_path=tokenizer_path,
        **kwargs,
    )