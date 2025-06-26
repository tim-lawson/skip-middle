import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
import wandb
from pandas.io.json._normalize import nested_to_record  # type: ignore
from simple_parsing import Serializable, parse

from projects.skip_middle.analysis.estimate_flops import flops_transformer
from projects.skip_middle.model import ModelConfig, get_model_config


def get_config(run: Any) -> tuple[ModelConfig, int]:
    try:
        batch_size = int(run.config["data"]["batch_size"])
    except KeyError:
        batch_size = int(run.config["batch_size"])
    try:
        model = run.config["model"]
    except KeyError:
        model = run.config
    config = get_model_config(model)
    return config, batch_size


def get_infer_flops(run: Any) -> int:
    config, _ = get_config(run)
    return flops_transformer(config, sparse=True, sparsity=get_zeros(run))


def get_train_flops(run: Any) -> int:
    config, batch_size = get_config(run)
    flops = 0
    for _, row in run.history(keys=["train/zeros/mean"]).iterrows():
        sparsity = row["train/zeros/mean"]
        flops += flops_transformer(config, sparse=True, sparsity=sparsity)
    if flops == 0:
        for _, _ in run.history(keys=["train/loss"]).iterrows():
            flops += flops_transformer(config, sparse=False)
    flops *= batch_size
    return flops


def get_loss(run: Any) -> float:
    val_loss = 0
    for _, row in run.history(keys=["val/loss"]).iterrows():
        val_loss = row["val/loss"]
    return val_loss


def get_zeros(run: Any) -> float:
    val_zeros = 0
    for _, row in run.history(keys=["val/zeros/mean"]).iterrows():
        val_zeros = row["val/zeros/mean"]
    return val_zeros


@dataclass
class Config(Serializable):
    project: str


if __name__ == "__main__":
    config = parse(Config)
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(outdir, exist_ok=True)

    api = wandb.Api()
    rows: list[dict] = []
    for project in api.projects():
        if project.name != config.project:
            continue
        print("project", project.name)
        runs = api.runs(f"timlawson-/{project.name}")
        for run in runs:
            try:
                loss = get_loss(run)
                infer_flops = get_infer_flops(run)
                train_flops = get_train_flops(run)
                zeros = get_zeros(run)
                print(
                    "run",
                    run.name,
                    f"infer: {infer_flops:.3e} /",
                    f"train: {train_flops:.3e} FLOPs,",
                    f"zeros: {zeros:.3f},",
                    f"loss: {loss:.3f}",
                )
                rows.append(
                    {
                        "project": project.name,
                        "run": run.name,
                        "infer_flops": infer_flops,
                        "train_flops": train_flops,
                        "zeros": zeros,
                        "loss": loss,
                        "state": run.state,
                        **nested_to_record(run.config, sep="."),
                        **run.summary,
                    }
                )
            except KeyError as e:
                print("KeyError", run.name, e)
                continue
    if len(rows) == 0:
        raise ValueError("No runs found for project.")
    pd.DataFrame(rows).to_csv(
        os.path.join(outdir, f"flops_loss_{config.project}.csv"), index=False
    )
