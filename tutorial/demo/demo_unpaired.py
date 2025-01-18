import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sCIN.utils.utils import(split_full_data,
                             extract_counts, 
                             read_config, 
                             make_plots,
                             make_unpaired,
                             make_unpaired_random)
from sCIN.models.sCIN import (Mod1Encoder, 
                              Mod2Encoder, 
                              sCIN, 
                              train_sCIN_unpaired, 
                              get_emb_sCIN, 
                              pca_with_nans)
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str)
    args = parser.parse_args()

    config = read_config(args.cfg_path)

    SETTINGS = config["SETTINGS"]
    BASE_SAVE_DIR = f"results/{args.data}/main_unpaired"
    epochs = SETTINGS.get("EPOCHS", 150)
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    prop = 0.2
    save_dir_p = os.path.join(save_dir_seed, f"p{int(prop * 100)}")
    os.makedirs(save_dir_p, exist_ok=True)

    mod1_path, mod2_path = DATA["PBMC"]
    mod1 = ad.read_h5ad(mod1_path)
    mod2 = ad.read_h5ad(mod2_path)
    print("Data loaded successfully.")

    save_dir_seed = os.path.join(BASE_SAVE_DIR, f"rep{SEED + 1}")
    os.makedirs(save_dir_seed, exist_ok=True)

    mod1_train, mod1_test, mod2_train, mod2_test, \
        labels_train, labels_test = split_full_data(mod1, mod2, seed=SEED)
    print("Data splitted!")

    mod1_train_unp, mod1_lbls_unp, mod2_train_unp, mod2_lbls_unp = \
        make_unpaired(mod1_train, mod2_train, labels_train, SEED, p=prop)
    
    train_dict = train_sCIN_unpaired(mod1_train_unp, mod2_train_unp, 
                                             [mod1_lbls_unp, mod2_lbls_unp], 
                                             epochs=epochs, settings=SETTINGS, 
                                             seed=SEED, save_dir=save_dir_p, 
                                             is_pca=True)
    
    mod1_embs, mod2_embs = get_emb_sCIN(mod1_test, mod2_test, 
                                                labels_test, train_dict, 
                                                save_dir=save_dir_p, seed=SEED,
                                                is_pca=True)
            
    recall_at_k, num_pairs, cell_type_acc, asw = assess(mod1_embs, mod2_embs, 
                                                        labels_test, seed=SEED)
    for k,v in recall_at_k:
        print(f"k = {int(k)} | Recall@k = {v}")
    
    print(f"Number of detected true pairs = {int(num_pairs)}")
    print(f"Cell type accuracy = {cell_type_acc}")
    print(f"Average Silouette Width = {asw}")


if __name__ == "__main__":
    main()