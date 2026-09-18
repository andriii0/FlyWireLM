from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from flylm.config import TrainingConfig
from flylm.data import (
    TINY_PATTERN,
    batchify,
    load_corpus_range,
    make_tiny_stream,
)
from flylm.generation import generate, generation_match
from flylm.model import FrozenFlyWireLM, load_flywire_model


def trainable_parameter_names(model: FrozenFlyWireLM) -> set[str]:
    return {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def assert_frozen_connectome(model: FrozenFlyWireLM) -> None:
    expected = {
        "embedding.weight",
        "input_projection.weight",
        "decoder.weight",
    }
    actual = trainable_parameter_names(model)
    if actual != expected:
        raise RuntimeError(f"unexpected trainable parameters: {sorted(actual)}")
    if model.adjacency.requires_grad:
        raise RuntimeError("connectome adjacency must be frozen")


def train_step(
    model: FrozenFlyWireLM,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    state: torch.Tensor,
) -> tuple[float, float, torch.Tensor]:
    optimizer.zero_grad(set_to_none=True)
    logits, state = model(inputs, state)
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    accuracy = (logits.argmax(dim=-1) == targets).float().mean()
    return float(loss.detach()), float(accuracy.detach()), state.detach()


@torch.no_grad()
def evaluate(
    model: FrozenFlyWireLM,
    stream: torch.Tensor,
    bptt: int,
) -> dict[str, float | int]:
    state = model.initial_state(stream.shape[1])
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for position in range(0, len(stream) - 1, bptt):
        length = min(bptt, len(stream) - position - 1)
        inputs = stream[position : position + length]
        targets = stream[position + 1 : position + length + 1]
        logits, state = model(inputs, state)
        flat_logits = logits.flatten(0, 1)
        flat_targets = targets.flatten()
        total_loss += float(
            F.cross_entropy(flat_logits, flat_targets, reduction="sum")
        )
        total_correct += int((logits.argmax(dim=-1) == targets).sum())
        total_tokens += flat_targets.numel()

    loss = total_loss / total_tokens
    return {
        "validation_loss": loss,
        "validation_perplexity": math.exp(loss),
        "validation_accuracy": total_correct / total_tokens,
        "validation_tokens": total_tokens,
    }


class FlyLmTrainer:
    def __init__(self, config: TrainingConfig) -> None:
        config.validate()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for full-connectome training")

        self.config = config
        self.device = torch.device("cuda")
        torch.manual_seed(config.seed)
        self.model = load_flywire_model(
            config.graph,
            self.device,
            config.embedding_dim,
            config.recurrent_steps,
            config.alpha,
            config.recurrent_scale,
        )
        assert_frozen_connectome(self.model)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )
        self.prior_training_seconds = 0.0
        self.prior_training_bytes = 0
        self._restore_checkpoint()

    def run(self) -> None:
        stream = self._training_stream()
        state = self.model.initial_state(self.config.batch_size)
        position = 0
        bytes_seen = 0
        started = time.perf_counter()
        log_path = self.config.output_dir / "loss.jsonl"
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        loss = 0.0
        accuracy = 0.0
        for step in range(1, self.config.max_steps + 1):
            if position + 1 >= len(stream):
                position = 0
                state = self.model.initial_state(self.config.batch_size)

            length = min(
                self.config.bptt,
                len(stream) - position - 1,
            )
            inputs = stream[position : position + length]
            targets = stream[position + 1 : position + length + 1]
            position += length
            bytes_seen += inputs.numel()
            loss, accuracy, state = train_step(
                self.model,
                self.optimizer,
                inputs,
                targets,
                state,
            )

            if step == 1 or step % self.config.log_every == 0:
                self._log_step(log_path, step, loss, accuracy)

            if self.config.corpus is None and accuracy >= self.config.target_accuracy:
                if self._tiny_overfit_succeeded(
                    step,
                    loss,
                    accuracy,
                    started,
                    bytes_seen,
                ):
                    return

        metrics = self._timing_metrics(started, bytes_seen)
        metrics.update(self._validation_metrics())
        generated, generation_accuracy = self._final_generation()
        print(generated.decode("utf-8", errors="replace"), flush=True)
        self._save_checkpoint(
            self.config.max_steps,
            loss,
            accuracy,
            generated,
            generation_accuracy,
            metrics,
        )
        if self.config.corpus is None:
            raise RuntimeError(
                "tiny overfit did not reach "
                f"{self.config.target_accuracy:.1%} accuracy"
            )

    def _restore_checkpoint(self) -> None:
        if self.config.checkpoint is None:
            return

        checkpoint = torch.load(
            self.config.checkpoint,
            map_location=self.device,
        )
        self.model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        previous = checkpoint.get("config", {})
        self.prior_training_seconds = float(
            previous.get(
                "total_training_seconds",
                previous.get("training_seconds", 0.0),
            )
        )
        self.prior_training_bytes = int(
            previous.get(
                "total_training_bytes_seen",
                previous.get("training_bytes_seen", 0),
            )
        )

    def _training_stream(self) -> torch.Tensor:
        if self.config.corpus is None:
            return make_tiny_stream(
                self.config.batch_size,
                self.config.bptt,
                self.device,
            )

        corpus = load_corpus_range(
            self.config.corpus,
            self.config.corpus_offset,
            self.config.corpus_bytes,
        )
        return batchify(corpus, self.config.batch_size, self.device)

    def _validation_metrics(self) -> dict[str, float | int]:
        if self.config.validation_corpus is None:
            return {}

        validation = load_corpus_range(
            self.config.validation_corpus,
            0,
            self.config.validation_bytes,
        )
        stream = batchify(
            validation,
            self.config.batch_size,
            self.device,
        )
        metrics = evaluate(self.model, stream, self.config.bptt)
        print(
            f"validation loss={metrics['validation_loss']:.4f} "
            f"perplexity={metrics['validation_perplexity']:.2f} "
            f"accuracy={metrics['validation_accuracy']:.4f}",
            flush=True,
        )
        return metrics

    def _tiny_overfit_succeeded(
        self,
        step: int,
        loss: float,
        accuracy: float,
        started: float,
        bytes_seen: int,
    ) -> bool:
        generated = generate(
            self.model,
            TINY_PATTERN[:8],
            len(TINY_PATTERN) * 4,
        )
        generated_accuracy = generation_match(generated)
        print(
            f"generation_accuracy={generated_accuracy:.4f}\n"
            f"{generated.decode('utf-8', errors='replace')}",
            flush=True,
        )
        if generated_accuracy < self.config.target_generation_accuracy:
            return False

        self._save_checkpoint(
            step,
            loss,
            accuracy,
            generated,
            generated_accuracy,
            self._timing_metrics(started, bytes_seen),
        )
        return True

    def _final_generation(self) -> tuple[bytes, float | None]:
        if self.config.corpus is None:
            generated = generate(
                self.model,
                TINY_PATTERN[:8],
                len(TINY_PATTERN) * 4,
            )
            return generated, generation_match(generated)

        generated = generate(
            self.model,
            self.config.generation_seed.encode("utf-8"),
            self.config.generation_length,
            self.config.temperature,
        )
        return generated, None

    def _timing_metrics(
        self,
        started: float,
        bytes_seen: int,
    ) -> dict[str, float | int]:
        elapsed = time.perf_counter() - started
        return {
            "training_seconds": elapsed,
            "training_bytes_seen": bytes_seen,
            "total_training_seconds": self.prior_training_seconds + elapsed,
            "total_training_bytes_seen": self.prior_training_bytes + bytes_seen,
        }

    @staticmethod
    def _log_step(
        path: Path,
        step: int,
        loss: float,
        accuracy: float,
    ) -> None:
        record = {"step": step, "loss": loss, "accuracy": accuracy}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"step={step:5d} loss={loss:.4f} accuracy={accuracy:.4f}",
            flush=True,
        )

    def _save_checkpoint(
        self,
        step: int,
        loss: float,
        accuracy: float,
        generation: bytes,
        generation_accuracy: float | None,
        metrics: dict[str, float | int],
    ) -> None:
        config = self.config.as_dict()
        config.update(
            {
                "initialized_from": (
                    str(self.config.checkpoint)
                    if self.config.checkpoint is not None
                    else None
                ),
                "step": step,
                "loss": loss,
                "accuracy": accuracy,
                "generation_accuracy": generation_accuracy,
                "generation_hex": generation.hex(),
                "generation_text": generation.decode(
                    "utf-8",
                    errors="replace",
                ),
            }
        )
        config.update(metrics)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": config,
            },
            self.config.output_dir / "checkpoint.pt",
        )
        result_path = self.config.output_dir / "result.json"
        result_path.write_text(
            json.dumps(config, indent=2),
            encoding="utf-8",
        )
