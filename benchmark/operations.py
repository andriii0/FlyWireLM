from __future__ import annotations

import torch


def csr_propagate(
    adjacency: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    return torch.sparse.mm(adjacency, state)


def edge_propagate(
    pre: torch.Tensor,
    post: torch.Tensor,
    values: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    messages = state.index_select(0, pre) * values.unsqueeze(1)
    output = torch.zeros_like(state)
    return output.index_add_(0, post, messages)


def correctness_test(device: torch.device) -> None:
    dense = torch.tensor(
        [[0.0, 0.5, 0.0], [0.25, 0.0, 0.75], [1.0, 0.0, 0.0]],
        device=device,
    )
    crow = torch.tensor([0, 1, 3, 4], dtype=torch.int64, device=device)
    col = torch.tensor([1, 0, 2, 0], dtype=torch.int64, device=device)
    values = torch.tensor([0.5, 0.25, 0.75, 1.0], device=device)
    post = torch.tensor([0, 1, 1, 2], dtype=torch.int64, device=device)
    state = torch.randn(3, 4, device=device)
    expected = dense @ state
    adjacency = torch.sparse_csr_tensor(
        crow,
        col,
        values,
        size=(3, 3),
        device=device,
        check_invariants=False,
    )

    torch.testing.assert_close(
        csr_propagate(adjacency, state),
        expected,
    )
    torch.testing.assert_close(
        edge_propagate(col, post, values, state),
        expected,
    )
