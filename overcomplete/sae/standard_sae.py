from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from torch import nn
from typing_extensions import override

from .sae import SAE, SAEConfig

@dataclass
class StandardSAEConfig(SAEConfig):
    @override
    @classmethod
    def architecture(cls) -> str:
        return "standard"


class StandardSAE(SAE):

    def __init__(self, cfg: StandardSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

    def initialize_weights(self) -> None:
        self.W_enc = nn.Parameter(torch.empty(self.d_in, self.d_sae))
        self.b_enc = nn.Parameter(torch.zeros(self.d_sae))

        self.W_dec = nn.Parameter(torch.empty(self.d_sae, self.d_in))
        self.b_dec = nn.Parameter(torch.zeros(self.d_in))

        nn.init.xavier_uniform_(self.W_enc)
        nn.init.xavier_uniform_(self.W_dec)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        hidden_pre = x @ self.W_enc + self.b_enc
        return self.activation_fn(hidden_pre)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        return feature_acts @ self.W_dec + self.b_dec
