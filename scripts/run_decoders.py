import os
import anndata as ad
import torch
import torch.nn as nn
from torch.optim import Adam
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
from sc_cool.models.decoders import GEXDecoder, train_val_gex_decoder, list_embs, get_gt
from sc_cool.utils.utils import read_config
import argparse
from typing import List


def get_search_space():
    """Defines the hyperparameter search space for tuning."""
    return {
        "num_layers": tune.choice([2, 4, 6]),
        "hidden_dims": tune.grid_search([[32], [64], [128], [256]]),
        "out_poisson": tune.choice([True, False]),
        "batch_norm": tune.choice([True, False]),
        "activation": tune.choice(["relu", "tanh", "leaky_relu"]),
        "dropout": tune.uniform(0.0, 0.5),
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([32, 64, 128]),
        "is_checkpoints": tune.choice([True, False]),
    }


def hyperparam_opt(model:str, save_dir:str, datasets:List) -> None:

    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=50,
        grace_period=5,
        reduction_factor=2,
    )

    search_space = get_search_space()

    ray.init(ignore_reinit_error=True)
    try:
        for data in datasets:
            tune.run(
                tune.with_parameters(
                    train_val_gex_decoder,
                    data=data,
                    model=model,
                    save_dir=save_dir
                ),
                name=f"gex_optimizer_{data.lower()}",
                config=search_space,
                scheduler=scheduler,
                num_samples=20,
                resources_per_trial={"cpu": 8, "gpu": 0.5},
                fail_fast=True,
            )
    except Exception as e:
        print(f"Error during hyperparameter optimization: {e}")
    finally:
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization or inference")
    parser.add_argument("--hyperparam_opt", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--model", type=str, help="Name of the model")
    parser.add_argument("--save_dir", type=str, help="Save directory")
    parser.add_argument("--datasets", type=str, nargs="+", default=["SHARE", "PBMC", "CITE"], 
                        help="List of datasets for optimization. Default: ['SHARE', 'PBMC', 'CITE'].")
    args = parser.parse_args()

    if args.hyperparam_opt:
        print("Starting hyperparameter optimization...")
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir, exist_ok=True)
            
        hyperparam_opt(model=args.model, 
                       save_dir=args.save_dir,
                       datasets=args.datasets)
        print("Hyperparameter optimization completed.")

if __name__ == "__main__":
    main()