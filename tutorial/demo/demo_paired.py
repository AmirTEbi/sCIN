"""
A demo script to show how sCIN can be used for paired PBMC dataset.

To run:
> cd sCIN
sCIN > python tutorial/demo/demo_paired.py --cfg_path "configs/sCIN/sCIN_pbmc.json"
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sCIN.utils.utils import(split_full_data,
                            extract_counts, 
                            read_config, 
                            make_plots,
                            get_func_name)
from sCIN.models.sCIN import (Mod1Encoder, 
                              Mod2Encoder, 
                              sCIN, 
                              train_sCIN,
                              get_emb_sCIN)
from sCIN.benchmarks.assess import (compute_metrics, assess)
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import time
import argparse
import os

MODEL = "sCIN"
DATA = {
    "PBMC":("data/10x/10x-Multiome-Pbmc10k-RNA.h5ad", 
            "data/10x/10x-Multiome-Pbmc10k-ATAC.h5ad")
}
SEED = 0  # Random seed for reproducibility

def main() -> None:

    # Get config file path from terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str)
    args = parser.parse_args()

    # General settings
    config = read_config(args.cfg_path)
    SETTINGS = config["SETTINGS"]
    BASE_SAVE_DIR = f"results/PBMC/sCIN/V2"
    epochs = SETTINGS.get("EPOCHS", 150)

    # Read data
    mod1_path, mod2_path = DATA["PBMC"]
    mod1 = ad.read_h5ad(mod1_path)
    mod2 = ad.read_h5ad(mod2_path)
    print("Data loaded successfully.")

    # Get training and evaluation function for sCIN
    train_func_name, get_emb_func_name = get_func_name(MODEL)
    train_func = globals().get(train_func_name)
    get_emb_func = globals().get(get_emb_func_name)

    # Split data to train and test set
    mod1_train, mod1_test, mod2_train, mod2_test, \
                labels_train, labels_test = split_full_data(mod1, mod2, seed=SEED)
    print("Data splitted.")

    print("Training has been started ...")
    save_dir_seed = os.path.join(BASE_SAVE_DIR, f"rep{int(SEED/10 + 1)}")
    os.makedirs(save_dir_seed, exist_ok=True)
    
    # Training
    train_dict = train_func(mod1_train, mod2_train, labels_train,
                            epochs=epochs, settings=SETTINGS,
                            seed=SEED, is_pca=True, save_dir=save_dir_seed)
    
    print("Evaluation has been started ...")

    # Generate embeddings
    mod1_embs, mod2_embs = get_emb_func(mod1_test, mod2_test, labels_test, 
                                        train_dict, save_dir_seed,
                                        seed=SEED, is_pca=True)
    print(f"Embeddings saved at {save_dir_seed}")
    # Evaluations
    recall_at_k, num_pairs, cell_type_acc, asw = assess(mod1_embs, mod2_embs, 
                                                        labels_test, seed=SEED)
    for k,v in recall_at_k:
        print(f"k = {int(k)} | Recall@k = {v}")
    
    print(f"Number of detected true pairs = {int(num_pairs)}")
    print(f"Cell type accuracy = {cell_type_acc}")
    print(f"Average Silouette Width = {asw}")


if __name__ == "__main__":
    main()