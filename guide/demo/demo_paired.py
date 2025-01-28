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
                            make_plots)
from sCIN.models.sCIN import (Mod1Encoder, 
                              Mod2Encoder, 
                              sCIN, 
                              train_sCIN,
                              get_emb_sCIN)
# Imports your model's functions here.
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


# Datasets and their paths. Add key:value in form of "data_name":(mod1_path, mod2_path), if needed. 
DATA = {
    "PBMC":("data/10x/10x-Multiome-Pbmc10k-RNA.h5ad", 
            "data/10x/10x-Multiome-Pbmc10k-ATAC.h5ad")
}


def main() -> None:

    # Get options from terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str)
    parser.add_argument("--model", type=str, help="The models name should be exactly the same as the name used in your training and evaluaiton function.")  
    parser.add_argument("--base_save_dir", type=str)
    args = parser.parse_args()

    # General settings
    seeds = [0]  # Random seed for reproducibility. Add more seeds if needed. 
    config = read_config(args.cfg_path)
    SETTINGS = config["SETTINGS"]
    epochs = SETTINGS.get("EPOCHS", 150)

    # Read data
    mod1_path, mod2_path = DATA["PBMC"]
    mod1 = ad.read_h5ad(mod1_path)
    mod2 = ad.read_h5ad(mod2_path)
    print("Data loaded successfully.")

    # Get training and evaluation function for the model
    train_func_name = f"train_{args.model}"
    get_emb_func_name = f"get_emb_{args.model}"
    train_func = globals().get(train_func_name)
    get_emb_func = globals().get(get_emb_func_name)
    if not train_func:
        raise ValueError(f"Training function was not imported or has a name distinct from the allowed name. 
                         Note that the model's name in the training function must be as same as the --model
                         value.")
    if not get_emb_func:
        raise ValueError(f"get_emb function was not imported or has a name distinct from the allowed name. 
                         Note that the model's name in the get_emb function must be as same as the --model
                         value.")

    for seed in seeds:
        # Split data to train and test set
        mod1_train, mod1_test, mod2_train, mod2_test, \
                    labels_train, labels_test = split_full_data(mod1, mod2, seed=seed)
        print("Data splitted.")

        print("Training has been started ...")
        save_dir_seed = os.path.join(args.base_save_dir, f"rep{seeds.index(seed) + 1}")
        os.makedirs(save_dir_seed, exist_ok=True)
        
        # Training
        train_dict = train_func(mod1_train, mod2_train, labels_train,
                                epochs=epochs, settings=SETTINGS,
                                seed=seed, is_pca=True, save_dir=save_dir_seed)
        
        print("Evaluation has been started ...")

        # Generate embeddings
        mod1_embs, mod2_embs = get_emb_func(mod1_test, mod2_test, labels_test, 
                                            train_dict, save_dir_seed,
                                            seed=seed, is_pca=True)
        mod1_df = pd.DataFrame(
            data = mod1_embs,
            columns = [f"dim_{i}" for i in range(mod1_embs.shape[1])],
            index = [f"cell_{i}" for i in range(mod1_embs.shape[0])]
        )
        mod2_df = pd.DataFrame(
            data = mod2_embs,
            columns = [f"dim_{i}" for i in range(mod2_embs.shape[1])],
            index = [f"cell_{i}" for i in range(mod2_embs.shape[0])]
        )
        print("First modality embeddings:")
        print(mod1_df[:10, :10])
        print("Second modality embeddings:")
        print(mod2_df[:10, :10])
        mod1_df.to_csv(os.path.join(save_dir_seed, "embs", f"mod1_embs_rep_{seeds.index(seed)+1}.csv"))
        mod2_df.to_csv(os.path.join(save_dir_seed, "embs", f"mod2_embs_rep_{seeds.index(seed)+1}.csv"))
        print(f"Embeddings saved at {save_dir_seed}")

        # Evaluations
        recall_at_k, num_pairs, cell_type_acc, asw = assess(mod1_embs, mod2_embs, 
                                                            labels_test, seed=seed)
        for k,v in recall_at_k:
            print(f"k = {int(k)} | Recall@k = {v}")
        
        print(f"Number of detected true pairs = {int(num_pairs)}")
        print(f"Cell type accuracy = {cell_type_acc}")
        print(f"Average Silouette Width = {asw}")


if __name__ == "__main__":
    main()