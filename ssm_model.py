from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Tuple
from ssm_intro import (
    PatchEmbedding, SSMBlock, GlobalPooling, RMSNorm,
    PatchEmbeddingIO, SSMBlockIO, GlobalPoolingIO, NormalizationIO
)

@dataclass
class TemporalPredictionIO:
    input_shape: Tuple[int, int, int, int]  # B, C, H, W
    output_shape: Tuple[int, int]           # B, output_dim
    hidden_dim: int
    state_size: int
    n_layers: int
    patch_size: int

class TemporalStateSpaceModel(nn.Module):
    def __init__(self, contract: TemporalPredictionIO):
        super().__init__()
        self.contract = contract
        B, C, H, W = contract.input_shape
        
        assert H % contract.patch_size == 0 and W % contract.patch_size == 0
        n_patches = (H // contract.patch_size) * (W // contract.patch_size)
        
        self.embedding = PatchEmbedding(PatchEmbeddingIO(
            input_shape=(B, C, H, W),
            output_shape=(B, n_patches, contract.hidden_dim),
            patch_count=n_patches
        ))
        
        self.time_in = nn.Sequential(
            nn.Linear(1, contract.hidden_dim),
            nn.SiLU(),
            nn.Linear(contract.hidden_dim, contract.hidden_dim)
        )
        
        self.time_out = nn.Sequential(
            nn.Linear(1, contract.hidden_dim),
            nn.SiLU(),
            nn.Linear(contract.hidden_dim, contract.hidden_dim)
        )
        
        self.blocks = nn.ModuleList([
            SSMBlock(SSMBlockIO(
                input_shape=(B, n_patches, contract.hidden_dim),
                state_size=contract.state_size,
                mlp_ratio=4.0
            )) for _ in range(contract.n_layers)
        ])
        
        self.norm = RMSNorm(NormalizationIO(contract.hidden_dim, 1e-6))
        self.pool = GlobalPooling(GlobalPoolingIO(
            input_shape=(B, n_patches, contract.hidden_dim),
            output_shape=(B, contract.hidden_dim)
        ))
        
        H_out, D_out = contract.output_shape
        self.head = nn.Sequential(
            nn.Linear(contract.hidden_dim, contract.hidden_dim),
            nn.SiLU(),
            nn.Linear(contract.hidden_dim, D_out)
        )

    def forward(
        self,
        x: torch.Tensor,
        t_input: torch.Tensor,
        t_predict: torch.Tensor
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        assert x.shape == self.contract.input_shape
        assert t_input.shape == (B, 1)
        assert t_predict.shape == (B, 1)
        
        x = self.embedding(x)
        
        t_in_embed = self.time_in(t_input)
        t_out_embed = self.time_out(t_predict)
        
        x = x + t_in_embed.unsqueeze(1)
        x = x + t_out_embed.unsqueeze(1) 
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = self.pool(x)
        x = self.head(x)
        
        assert x.shape == self.contract.output_shape
        return x
