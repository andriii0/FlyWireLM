from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    graph: Path
    output_dir: Path
    checkpoint: Path | None = None
    corpus: Path | None = None
    corpus_offset: int = 0
    corpus_bytes: int = 0
    validation_corpus: Path | None = None
    validation_bytes: int = 0
    generation_seed: str = " = "
    generation_length: int = 400
    temperature: float = 0.8
    embedding_dim: int = 64
    recurrent_steps: int = 1
    alpha: float = 0.5
    recurrent_scale: float = 0.9
    batch_size: int = 16
    bptt: int = 32
    learning_rate: float = 1e-3
    max_steps: int = 2000
    target_accuracy: float = 0.995
    target_generation_accuracy: float = 0.95
    log_every: int = 20
    seed: int = 7

    def validate(self) -> None:
        positive_values = {
            "embedding_dim": self.embedding_dim,
            "recurrent_steps": self.recurrent_steps,
            "batch_size": self.batch_size,
            "bptt": self.bptt,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "log_every": self.log_every,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.corpus_offset < 0 or self.corpus_bytes < 0:
            raise ValueError("corpus ranges cannot be negative")
        if self.validation_bytes < 0:
            raise ValueError("validation_bytes cannot be negative")
        if self.generation_length < 0:
            raise ValueError("generation_length cannot be negative")

    def as_dict(self) -> dict[str, object]:
        values = asdict(self)
        for name in (
            "graph",
            "output_dir",
            "checkpoint",
            "corpus",
            "validation_corpus",
        ):
            value = values[name]
            values[name] = str(value) if value is not None else None
        return values
