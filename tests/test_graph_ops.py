import unittest

import torch

from benchmark.operations import correctness_test
from flylm.model import FrozenFlyWireLM


class GraphOperationTests(unittest.TestCase):
    def test_cpu_sparse_implementations_match_dense(self) -> None:
        correctness_test(torch.device("cpu"))

    def test_model_has_only_external_trainable_parameters(self) -> None:
        crow = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        col = torch.tensor([0, 1, 2], dtype=torch.int64)
        values = torch.ones(3)
        adjacency = torch.sparse_csr_tensor(
            crow,
            col,
            values,
            size=(3, 3),
            check_invariants=False,
        )
        model = FrozenFlyWireLM(
            adjacency=adjacency,
            afferent_indices=torch.tensor([0]),
            efferent_indices=torch.tensor([2]),
            embedding_dim=4,
            recurrent_steps=1,
            alpha=0.5,
            recurrent_scale=0.9,
        )
        names = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(
            names,
            {
                "embedding.weight",
                "input_projection.weight",
                "decoder.weight",
            },
        )
        self.assertFalse(model.adjacency.requires_grad)
        logits, state = model(torch.tensor([[1], [2]]))
        self.assertEqual(logits.shape, (2, 1, 256))
        self.assertEqual(state.shape, (3, 1))


if __name__ == "__main__":
    unittest.main()
