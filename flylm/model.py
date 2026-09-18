from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class FrozenFlyWireLM(nn.Module):
    def __init__(
        self,
        adjacency: torch.Tensor,
        afferent_indices: torch.Tensor,
        efferent_indices: torch.Tensor,
        embedding_dim: int,
        recurrent_steps: int,
        alpha: float,
        recurrent_scale: float,
    ) -> None:
        super().__init__()
        if recurrent_steps < 1:
            raise ValueError("recurrent_steps must be positive")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if recurrent_scale <= 0.0:
            raise ValueError("recurrent_scale must be positive")

        self.node_count = adjacency.size(0)
        self.recurrent_steps = recurrent_steps
        self.alpha = float(alpha)
        self.recurrent_scale = float(recurrent_scale)
        self.register_buffer("adjacency", adjacency, persistent=False)
        self.register_buffer(
            "afferent_indices", afferent_indices, persistent=False
        )
        self.register_buffer(
            "efferent_indices", efferent_indices, persistent=False
        )

        self.embedding = nn.Embedding(256, embedding_dim)
        self.input_projection = nn.Linear(
            embedding_dim, len(afferent_indices), bias=False
        )
        self.decoder = nn.Linear(len(efferent_indices), 256, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.5)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

    def initial_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            self.node_count,
            batch_size,
            device=self.embedding.weight.device,
            dtype=self.embedding.weight.dtype,
        )

    def recurrent_step(
        self, state: torch.Tensor, afferent_drive: torch.Tensor | None
    ) -> torch.Tensor:
        preactivation = self.recurrent_scale * torch.sparse.mm(
            self.adjacency, state
        )
        if afferent_drive is not None:
            preactivation = torch.index_add(
                preactivation,
                0,
                self.afferent_indices,
                afferent_drive.transpose(0, 1),
            )
        candidate = torch.tanh(preactivation)
        return (1.0 - self.alpha) * state + self.alpha * candidate

    def forward(
        self, tokens: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [time, batch]")
        if state is None:
            state = self.initial_state(tokens.shape[1])

        logits: list[torch.Tensor] = []
        for token_ids in tokens:
            afferent_drive = self.input_projection(self.embedding(token_ids))
            for step in range(self.recurrent_steps):
                state = self.recurrent_step(
                    state, afferent_drive if step == 0 else None
                )
            efferent_state = torch.index_select(
                state, 0, self.efferent_indices
            ).transpose(0, 1)
            logits.append(self.decoder(efferent_state))
        return torch.stack(logits), state


def load_flywire_model(
    graph_path: Path,
    device: torch.device,
    embedding_dim: int,
    recurrent_steps: int,
    alpha: float,
    recurrent_scale: float,
) -> FrozenFlyWireLM:
    graph = np.load(graph_path)
    crow = torch.from_numpy(graph["crow_indices"]).to(device)
    col = torch.from_numpy(graph["col_indices"]).to(device)
    values = torch.from_numpy(graph["values"]).to(device)
    node_count = len(graph["root_ids"])
    adjacency = torch.sparse_csr_tensor(
        crow,
        col,
        values,
        size=(node_count, node_count),
        device=device,
        check_invariants=False,
    )
    afferent = torch.from_numpy(graph["afferent_indices"]).to(device)
    efferent = torch.from_numpy(graph["efferent_indices"]).to(device)
    return FrozenFlyWireLM(
        adjacency=adjacency,
        afferent_indices=afferent,
        efferent_indices=efferent,
        embedding_dim=embedding_dim,
        recurrent_steps=recurrent_steps,
        alpha=alpha,
        recurrent_scale=recurrent_scale,
    ).to(device)
