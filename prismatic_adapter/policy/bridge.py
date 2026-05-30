"""Bridge Attention action policy.

The block mirrors VLA-Adapter's core idea while keeping the implementation
backbone-independent: a small action decoder attends to its own action-time
latents, ActionQuery states, optional proprioception, and Raw visual tokens.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ProprioProjector(nn.Module):
    """Project robot state into the language-model hidden space."""

    def __init__(self, proprio_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proprio_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.net(proprio)


class RotaryEmbedding(nn.Module):
    """Small RoPE helper for Bridge attention."""

    def __init__(self, head_dim: int, base: int = 10000) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, length: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(length, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (_rotate_half(x) * sin)


class BridgeAttentionBlock(nn.Module):
    """A transformer-style block with self, ActionQuery, and Raw branches."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_rope: bool = True,
        gate_raw_branch: bool = True,
        ffn_multiplier: int = 4,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_rope = use_rope
        self.gate_raw_branch = gate_raw_branch

        self.q_norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_self = nn.Linear(hidden_size, hidden_size)
        self.v_self = nn.Linear(hidden_size, hidden_size)
        self.k_aq = nn.Linear(hidden_size, hidden_size)
        self.v_aq = nn.Linear(hidden_size, hidden_size)
        self.k_raw = nn.Linear(hidden_size, hidden_size)
        self.v_raw = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        self.raw_gate = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim) if use_rope else None

        ffn_hidden = hidden_size * ffn_multiplier
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_size),
        )

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, length, _ = x.shape
        return x.view(bsz, length, self.num_heads, self.head_dim).transpose(1, 2)

    def _project_memory(
        self,
        x: torch.Tensor,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._heads(k_proj(x)), self._heads(v_proj(x))

    def forward(
        self,
        x: torch.Tensor,
        raw_tokens: torch.Tensor,
        action_query_tokens: torch.Tensor,
        proprio_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if proprio_token is not None:
            action_query_tokens = torch.cat([action_query_tokens, proprio_token], dim=1)

        residual = x
        x_norm = self.q_norm(x)
        q = self._heads(self.q_proj(x_norm))
        k_self, v_self = self._project_memory(x_norm, self.k_self, self.v_self)
        k_aq, v_aq = self._project_memory(action_query_tokens, self.k_aq, self.v_aq)
        k_raw, v_raw = self._project_memory(raw_tokens, self.k_raw, self.v_raw)

        if self.rope is not None:
            cos, sin = self.rope(q.shape[-2], x.device, x.dtype)
            q = apply_rope(q, cos, sin)
            k_self = apply_rope(k_self, cos, sin)
            cos, sin = self.rope(k_aq.shape[-2], x.device, x.dtype)
            k_aq = apply_rope(k_aq, cos, sin)
            cos, sin = self.rope(k_raw.shape[-2], x.device, x.dtype)
            k_raw = apply_rope(k_raw, cos, sin)

        scores = [
            torch.matmul(q, k_self.transpose(-2, -1)),
            torch.matmul(q, k_aq.transpose(-2, -1)),
        ]
        raw_scores = torch.matmul(q, k_raw.transpose(-2, -1))
        if self.gate_raw_branch:
            raw_scores = raw_scores * torch.tanh(self.raw_gate)
        scores.append(raw_scores)

        attn_scores = torch.cat(scores, dim=-1) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        values = torch.cat([v_self, v_aq, v_raw], dim=2)
        attended = torch.matmul(attn_weights, values)
        attended = attended.transpose(1, 2).contiguous().view_as(x)
        x = residual + self.dropout(self.o_proj(attended))
        return x + self.dropout(self.ffn(x))


class BridgeActionHead(nn.Module):
    """Map VLM layer conditions to continuous action chunks."""

    def __init__(
        self,
        hidden_size: int,
        action_dim: int,
        action_horizon: int,
        num_layers: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_rope: bool = True,
        gate_raw_branch: bool = True,
        ffn_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_layers = num_layers

        self.action_latents = nn.Parameter(torch.zeros(action_horizon, hidden_size))
        nn.init.normal_(self.action_latents, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList(
            [
                BridgeAttentionBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_rope=use_rope,
                    gate_raw_branch=gate_raw_branch,
                    ffn_multiplier=ffn_multiplier,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, action_dim)

    def forward(
        self,
        raw_tokens: torch.Tensor,
        action_query_tokens: torch.Tensor,
        proprio_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict `[B, H, A]` normalized actions.

        Args:
            raw_tokens: `[B, L, R, D]` per-layer Raw visual tokens.
            action_query_tokens: `[B, L, Q, D]` per-layer AQ states.
            proprio_token: optional `[B, 1, D]`.
        """

        if raw_tokens.ndim != 4 or action_query_tokens.ndim != 4:
            raise ValueError("raw_tokens and action_query_tokens must be [B, L, T, D]")
        if raw_tokens.shape[:2] != action_query_tokens.shape[:2]:
            raise ValueError("raw_tokens and action_query_tokens must use the same batch/layer dims")
        if raw_tokens.shape[-1] != self.hidden_size:
            raise ValueError("condition hidden size does not match policy hidden size")

        batch_size, available_layers = raw_tokens.shape[:2]
        x = self.action_latents.unsqueeze(0).expand(batch_size, -1, -1)
        for idx, block in enumerate(self.blocks):
            source_idx = min(idx, available_layers - 1)
            x = block(
                x,
                raw_tokens=raw_tokens[:, source_idx],
                action_query_tokens=action_query_tokens[:, source_idx],
                proprio_token=proprio_token,
            )
        return self.out_proj(self.out_norm(x))
