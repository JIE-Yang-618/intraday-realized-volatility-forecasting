from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import yaml

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@dataclass
class ProjectConfig:
    name: str
    seed: int
    output_dir: str

@dataclass
class DataConfig:
    kind: str
    n_assets: int
    n_clusters: int
    t_train: int
    t_val: int
    t_test: int
    horizon_steps: int
    lag_p: int

@dataclass
class SchemeConfig:
    name: str
    use_market_rv: bool = False
    use_cluster_rv: bool = False

@dataclass
class ModelConfig:
    name: str
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 64

@dataclass
class MSAConfig:
    horizons_days: List[int]
    obs_per_day: int
    conv_channels: int = 8
    attn_size: int = 16

@dataclass
class EvalConfig:
    metrics: List[str]
    sharpe_ratio: float = 0.4
    risk_aversion: float = 2.0
    tc_rate: float = 0.0

@dataclass
class ExperimentConfig:
    project: ProjectConfig
    data: DataConfig
    scheme: SchemeConfig
    model: ModelConfig
    msa: Optional[MSAConfig]
    eval: EvalConfig

def parse_config(d: Dict[str, Any]) -> ExperimentConfig:
    proj = ProjectConfig(**d["project"])
    dat = DataConfig(**d["data"])
    sch = SchemeConfig(**d["scheme"])
    mod = ModelConfig(**d["model"])
    msa = None
    if "msa" in d and d["msa"] is not None:
        msa = MSAConfig(**d["msa"])
    ev = EvalConfig(**d["eval"])
    return ExperimentConfig(project=proj, data=dat, scheme=sch, model=mod, msa=msa, eval=ev)
