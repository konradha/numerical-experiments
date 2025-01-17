from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, NamedTuple

@dataclass
class PatchEmbeddingIO:
    input_shape: Tuple[int, int, int, int]  # B, C, H, W
    output_shape: Tuple[int, int, int]      # B, N, D
    patch_count: int                        # H*W/P^2
    
@dataclass
class SSMStateIO:
    batch_size: int
    sequence_length: int
    hidden_dim: int
    state_size: int

@dataclass
class PositionalEncodingIO:
    input_shape: Tuple[int, int, int]       # B, L, D
    max_sequence_length: int
    
@dataclass
class NormalizationIO:
    feature_dim: int
    eps: float

@dataclass
class MLPIO:
    input_dim: int
    hidden_dim: int
    output_dim: int
    
@dataclass
class GlobalPoolingIO:
    input_shape: Tuple[int, int, int]       # B, L, D
    output_shape: Tuple[int, int]           # B, D
    
@dataclass
class SSMBlockIO:
    input_shape: Tuple[int, int, int]       # B, L, D
    state_size: int
    mlp_ratio: float

class PatchEmbedding(nn.Module):
    def __init__(self, contract: PatchEmbeddingIO):
        super().__init__()
        B, C, H, W = contract.input_shape
        _, N, D = contract.output_shape
        P = int(math.sqrt((H * W) / contract.patch_count))
        
        self.proj = nn.Conv2d(C, D, kernel_size=P, stride=P)
        self.norm = RMSNorm(NormalizationIO(D, 1e-6))
        self.contract = contract

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert (B, C, H, W) == self.contract.input_shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        assert x.shape == self.contract.output_shape
        return x

class LinearStateSpaceLayer(nn.Module):
    def __init__(self, contract: SSMStateIO):
        super().__init__()
        self.contract = contract
        
        log_dt = torch.linspace(math.log(0.001), math.log(0.1), contract.state_size)
        self.register_buffer('dt', torch.exp(log_dt))
        
        # A stable -> Im{\lambda_i} < 0 for all eigenvalues of A
        self.A = nn.Parameter(torch.randn(contract.hidden_dim, contract.state_size))
        self.B = nn.Parameter(torch.randn(contract.hidden_dim, contract.state_size))
        self.C = nn.Parameter(torch.randn(contract.hidden_dim, contract.state_size))
        self.D = nn.Parameter(torch.zeros(contract.hidden_dim))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        B, L, D = u.shape
        assert B == self.contract.batch_size
        assert L == self.contract.sequence_length
        assert D == self.contract.hidden_dim
        
        A = -torch.exp(self.A.unsqueeze(1))
        dt = self.dt.view(1, 1, -1)
        dA = torch.exp(A * dt)
        dB = self.B.unsqueeze(1) * (1 - dA) / (-A)
        
        x = torch.zeros(B, D, self.contract.state_size, device=u.device)
        out = torch.zeros_like(u)
        
        for i in range(L):
            x = x * dA + u[:, i:i+1].transpose(1, 2).unsqueeze(-1) * dB
            out[:, i] = (x * self.C.unsqueeze(0).unsqueeze(-1)).sum(dim=(2, 3))
        
        out = out + u * self.D.unsqueeze(0).unsqueeze(1)
        assert out.shape == (B, L, D)
        return out

class RMSNorm(nn.Module):
    def __init__(self, contract: NormalizationIO):
        super().__init__()
        self.contract = contract
        self.scale = nn.Parameter(torch.ones(contract.feature_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.contract.feature_dim
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.contract.eps)
        return x * norm * self.scale

class LayerNorm(nn.Module):
    def __init__(self, contract: NormalizationIO):
        super().__init__()
        self.contract = contract
        self.g = nn.Parameter(torch.ones(contract.feature_dim))
        self.b = nn.Parameter(torch.zeros(contract.feature_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.contract.feature_dim
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.contract.eps) * self.g + self.b

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, contract: PositionalEncodingIO):
        super().__init__()
        self.contract = contract
        _, L, D = contract.input_shape
        self.pe = nn.Parameter(torch.randn(1, contract.max_sequence_length, D) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        assert (B, L, D) == self.contract.input_shape
        assert L <= self.contract.max_sequence_length
        return x + self.pe[:, :L]

class MLP(nn.Module):
    def __init__(self, contract: MLPIO):
        super().__init__()
        self.contract = contract
        self.net = nn.Sequential(
            nn.Linear(contract.input_dim, contract.hidden_dim),
            nn.GELU(),
            nn.Linear(contract.hidden_dim, contract.output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.contract.input_dim
        x = self.net(x)
        assert x.shape[-1] == self.contract.output_dim
        return x

class GlobalPooling(nn.Module):
    def __init__(self, contract: GlobalPoolingIO):
        super().__init__()
        self.contract = contract

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        assert (B, L, D) == self.contract.input_shape
        x = x.mean(dim=1)
        assert x.shape == self.contract.output_shape
        return x

class SSMBlock(nn.Module):
    def __init__(self, contract: SSMBlockIO):
        super().__init__()
        B, L, D = contract.input_shape
        
        self.norm1 = RMSNorm(NormalizationIO(D, 1e-6))
        self.ssm = LinearStateSpaceLayer(SSMStateIO(B, L, D, contract.state_size))
        self.norm2 = RMSNorm(NormalizationIO(D, 1e-6))
        self.mlp = MLP(MLPIO(D, int(D * contract.mlp_ratio), D))
        self.contract = contract

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape == self.contract.input_shape
        x = x + self.ssm(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        assert x.shape == self.contract.input_shape
        return x
