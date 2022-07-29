from dataclasses import dataclass, field
from typing import Optional
import yaml
import os


DISCOURSE_TYPES = [
    "Lead",
    "Position",
    "Claim",
    "Counterclaim",
    "Rebuttal",
    "Evidence",
    "Concluding Statement",
]

BIO_LABELS = (
    ["O"]
    + [f"B-{t}" for t in DISCOURSE_TYPES]
    + [f"I-{t}" for t in DISCOURSE_TYPES]
)

LABEL2ID = {t: i for i, t in enumerate(DISCOURSE_TYPES)}
ID2LABEL = {i: t for i, t in enumerate(DISCOURSE_TYPES)}
BIO2ID = {t: i for i, t in enumerate(BIO_LABELS)}
ID2BIO = {i: t for i, t in enumerate(BIO_LABELS)}


@dataclass
class ModelConfig:
    encoder: str = "bert-base-uncased"
    use_crf: bool = False
    use_sentence_encoder: bool = False
    use_hde: bool = False
    label_scheme: str = "category"
    max_seq_len: int = 512
    num_segments: int = 4
    segment_len: int = 256
    max_sentences: int = 100
    sentence_encoder_layers: int = 2
    sentence_encoder_heads: int = 8
    hidden_dim: int = 768
    dropout: float = 0.1


@dataclass
class TrainConfig:
    learning_rate: float = 2e-5
    batch_size: int = 8
    n_epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    patience: int = 5
    seed: int = 42


@dataclass
class DataConfig:
    train_file: str = "data/train.csv"
    dev_file: str = "data/dev.csv"
    test_file: str = "data/test.csv"
    essay_dir: str = "data/train"


@dataclass
class DiscourseConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = "checkpoints/"

    @classmethod
    def from_yaml(cls, path: str) -> "DiscourseConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        model_cfg = ModelConfig(**raw.get("model", {}))
        train_cfg = TrainConfig(**raw.get("train", {}))
        data_cfg = DataConfig(**raw.get("data", {}))
        output_dir = raw.get("output_dir", "checkpoints/")

        return cls(model=model_cfg, train=train_cfg, data=data_cfg, output_dir=output_dir)
