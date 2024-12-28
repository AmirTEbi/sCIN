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
from sc_cool.models.decoders import (GEXDecoder, 
                                     train_val_gex_decoder, 
                                     list_embs, get_gt, 
                                     get_search_space, 
                                     hyperparam_opt )
from sc_cool.utils.utils import read_config
import argparse
from typing import Dict, Tuple, List
import traceback


def main():

    DEFAULT_DATASETS = ["SHARE", "PBMC", "CITE"]

    parser = argparse.ArgumentParser(description="Run hyperparameter optimization or inference")
    parser.add_argument("--hyperparam_opt", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--model", type=str, help="Name of the model")
    parser.add_argument("--save_dir", type=str, help="Save directory")
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=DEFAULT_DATASETS,
        help=f"List of datasets for optimization. Default: {DEFAULT_DATASETS}.")
    
    args = parser.parse_args()

    if args.hyperparam_opt:
        print("Starting hyperparameter optimization...")
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir, exist_ok=True)

        hyperparam_opt(
            model=args.model,
            save_dir=args.save_dir,
            datasets=args.datasets,
        )
        print("Hyperparameter optimization completed.")


if __name__ == "__main__":
    main()