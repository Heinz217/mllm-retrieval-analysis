from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from torch import nn
from jaxtyping import Float
from typing_extensions import override


@dataclass
class SAEConfig(ABC):

    d_in: int   # input dim
    d_sae: int  # hidden dim

    @classmethod
    @abstractmethod
    def architecture(cls) -> str:
        ...


class SAE(nn.Module, ABC):


    def __init__(self, cfg: SAEConfig, use_error_term: bool = False):
        super().__init__()
        self.cfg = cfg
        self.d_in = cfg.d_in
        self.d_sae = cfg.d_sae
        self.use_error_term = use_error_term

        # Shared parameters
        self.W_enc: nn.Parameter
        self.W_dec: nn.Parameter
        self.b_dec: nn.Parameter
        self.activation_fn = nn.ReLU()

        self.initialize_weights()

    @abstractmethod
    def initialize_weights(self) -> None:
        ...

    @abstractmethod
    def encode(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        ...

    @abstractmethod
    def decode(
        self, feature_acts: Float[torch.Tensor, "... d_sae"]
    ) -> Float[torch.Tensor, "... d_in"]:
        ...

    def forward(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> tuple[Float[torch.Tensor, "... d_in"], Float[torch.Tensor, "... d_sae"]]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
