import abc
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer


class BaseMinecraftLM(nn.Module, metaclass=abc.ABCMeta):

    PRETRAINED_LM_PATH = ""
    PRETRAINED_TOKENIZER_PATH = ""

    BASE_LM_PATH = ""
    BASE_TOKENIZER_PATH = ""

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
        self.load_pretrained(
            lm_path, tokenizer_path, lm, tokenizer, lm_kwargs, tokenizer_kwargs
        )
        self.context_length = context_length

    def train(self):
        self.lm.train()

    def eval(self):
        self.lm.eval()

    @property
    def device(self):
        return self.lm.device

    # def to(self, device: torch.device):
    #     self.lm = self.lm.to(device)
    #     return self

    # def save_model(self, checkpoint_path: str, it: int):
    #     self.lm.save_pretrained(os.path.join(checkpoint_path, f"iteration_{it}"))

    def save_model(self, checkpoint_path: str, it: int):
        save_dir = os.path.join(checkpoint_path, f"iteration_{it}")
        os.makedirs(save_dir, exist_ok=True)
        # 保存整个模型的权重，包含自定义 Embedding
        torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        # 同时保存配置和分词器
        self.lm.config.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

    @abc.abstractmethod
    def load_pretrained_lm(
        self, path: str, lm_kwargs: Dict[str, Any]
    ) -> PreTrainedModel:
        """
        Model to be used in level tile prediction
        """

    @abc.abstractmethod
    def load_pretrained_tokenizer(
        self, path: str, tokenizer_kwargs: Dict[str, Any]
    ) -> PreTrainedTokenizer:
        """
        Tokenizer to be used to read / decode levels
        """

    def load_pretrained(
        self,
        lm_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        lm: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        lm_kwargs: Dict[str, Any] = {},
        tokenizer_kwargs: Dict[str, Any] = {},
    ):
        if lm is None:
            if lm_path is None:
                lm_path = self.PRETRAINED_LM_PATH

            print(f"Using {lm_path} lm")
            self.lm = self.load_pretrained_lm(lm_path, lm_kwargs)

        if tokenizer is None:
            if tokenizer_path is None:
                tokenizer_path = self.PRETRAINED_LM_PATH

            print(f"Using {tokenizer_path} tokenizer")
            self.tokenizer = self.load_pretrained_tokenizer(
                tokenizer_path, tokenizer_kwargs
            )