from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from src.config import load_yaml, parse_config
from src.utils.seed import set_global_seed
from src.utils.logging import get_logger, ensure_dir
from src.data.toy import generate_toy_rv, make_splits
from src.models.factory import make_model
from src.training.runner import train_dispatch, predict_under_scheme
from src.eval.metrics import mse, mae, qlike, realized_utility, realized_utility_tc

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = parse_config(load_yaml(args.config))
    logger = get_logger("toy")
    set_global_seed(cfg.project.seed)
    out_dir = ensure_dir(cfg.project.output_dir)

    # Data
    T_total = cfg.data.t_train + cfg.data.t_val + cfg.data.t_test + cfg.data.lag_p + cfg.data.horizon_steps + 5
    ds = generate_toy_rv(
        seed=cfg.project.seed,
        T=T_total,
        N=cfg.data.n_assets,
        G=cfg.data.n_clusters,
    )
    (tr_a, tr_b), (va_a, va_b), (te_a, te_b) = make_splits(ds, cfg.data.t_train + cfg.data.lag_p + cfg.data.horizon_steps, cfg.data.t_val, cfg.data.t_test)

    # Model
    model_cfg = {
        "name": cfg.model.name,
        "hidden_size": cfg.model.hidden_size,
        "num_layers": cfg.model.num_layers,
        "dropout": cfg.model.dropout,
        "lr": cfg.model.lr,
        "epochs": cfg.model.epochs,
        "batch_size": cfg.model.batch_size,
        "seq_len": cfg.data.lag_p if cfg.data.lag_p <= 80 else 60,
    }
    msa_cfg = None
    if cfg.msa is not None:
        msa_cfg = {
            "horizons_days": cfg.msa.horizons_days,
            "obs_per_day": cfg.msa.obs_per_day,
            "conv_channels": cfg.msa.conv_channels,
            "attn_size": cfg.msa.attn_size,
        }

    def make() -> object:
        return make_model(model_cfg, msa_cfg)

    # Train under scheme
    scheme = cfg.scheme.name.lower()
    artifacts = train_dispatch(
        scheme=scheme,
        make_model=make,
        rv=ds.rv,
        rv_mkt=ds.rv_mkt if cfg.scheme.use_market_rv else None,
        rv_clu=ds.rv_clu if cfg.scheme.use_cluster_rv else None,
        cluster_id=ds.cluster_id if scheme == "cluster" else None,
        p=cfg.data.lag_p,
        h=cfg.data.horizon_steps,
        split=(tr_a, tr_b),
    )

    # Predict
    preds = predict_under_scheme(
        artifacts=artifacts,
        scheme=scheme,
        rv=ds.rv,
        rv_mkt=ds.rv_mkt if cfg.scheme.use_market_rv else None,
        rv_clu=ds.rv_clu if cfg.scheme.use_cluster_rv else None,
        cluster_id=ds.cluster_id if scheme == "cluster" else None,
        p=cfg.data.lag_p,
        h=cfg.data.horizon_steps,
        split=(te_a, te_b),
    )

    # Aggregate metrics
    y_true_all = np.concatenate([p.y_true for p in preds.values()])
    y_pred_all = np.concatenate([p.y_pred for p in preds.values()])

    report = {
        "config": args.config,
        "scheme": scheme,
        "model": cfg.model.name,
        "mse": mse(y_true_all, y_pred_all),
        "mae": mae(y_true_all, y_pred_all),
        "qlike": qlike(y_true_all, y_pred_all),
        "ru": realized_utility(y_true_all, y_pred_all, cfg.eval.sharpe_ratio, cfg.eval.risk_aversion),
        "ru_tc": realized_utility_tc(y_true_all, y_pred_all, cfg.eval.tc_rate, cfg.eval.sharpe_ratio, cfg.eval.risk_aversion),
        "n_assets": cfg.data.n_assets,
    }

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote report to %s", out_dir / "report.json")
    logger.info("Summary: %s", report)

if __name__ == "__main__":
    main()
