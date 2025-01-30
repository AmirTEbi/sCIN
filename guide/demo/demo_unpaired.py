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

# Import your models functions here.

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
import logging as log

# Datasets and their paths. Add in form of "data_name":(mod1_path, mod2_path), if needed. 
DATA = {
    "PBMC":("data/10x/10x-Multiome-Pbmc10k-RNA.h5ad", 
            "data/10x/10x-Multiome-Pbmc10k-ATAC.h5ad")
}

log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--base_save_dir", type=str)
    args = parser.parse_args()

    config = read_config(args.cfg_path)

    seeds = [0]  # Random seed for reproducibility. Add more if needed.
    props = [0.2]  # The proportion of cells to be kept in the second modality. Add more if needed. 
    SETTINGS = config["SETTINGS"]
    epochs = SETTINGS.get("EPOCHS", 150)
    os.makedirs(args.base_save_dir, exist_ok=True)
   

    # Get training and evaluation function for the model
    train_func_name = f"train_{args.model}_unpaired"
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

    # Read dataß
    mod1_path, mod2_path = DATA["PBMC"]
    mod1 = ad.read_h5ad(mod1_path)
    mod2 = ad.read_h5ad(mod2_path)
    log.info("Data loaded.")

    for seed in seeds:
        rep = seeds.index(seed) + 1
        log.info(f"Replication {rep}")
        for p in props:
            # Define save directory
            save_dir_seed = os.path.join(args.base_save_dir, f"rep{rep}")
            os.makedirs(save_dir_seed, exist_ok=True)
            save_dir_p = [os.path.join(save_dir_seed, f"p{int(p * 100)}") for p in props]
            os.makedirs(save_dir_p, exist_ok=True)
            mod1_train, mod1_test, mod2_train, mod2_test, \
                labels_train, labels_test = split_full_data(mod1, mod2, seed=seed)
            log.info("Data splitted.")

            mod1_train_unp, mod1_lbls_unp, mod2_train_unp, mod2_lbls_unp = \
                make_unpaired(mod1_train, mod2_train, labels_train, seed, p=p)
            
            
            train_dict = train_func(mod1_train_unp, mod2_train_unp, 
                                    [mod1_lbls_unp, mod2_lbls_unp], 
                                    epochs=epochs, settings=SETTINGS, 
                                    seed=seed, save_dir=save_dir_p, 
                                    is_pca=True)
            log.info("Training finished.")
            
            mod1_embs, mod2_embs = get_emb_func(mod1_test, mod2_test, 
                                                labels_test, train_dict, 
                                                save_dir=save_dir_p, seed=seed,
                                                is_pca=True)
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

            log.info(f"First modality embeddings for replication {rep} :")
            print(mod1_df[:10, :10])
            log.info(f"Second modality embeddings for replication {rep}:")
            print(mod2_df[:10, :10])
            mod1_df.to_csv(os.path.join(save_dir_p, "embs", f"mod1_embs_rep_{rep}.csv"))
            mod2_df.to_csv(os.path.join(save_dir_p, "embs", f"mod2_embs_rep_{rep}.csv"))
            log.info(f"Embeddings saved at {save_dir_seed}")
        
            recall_at_k, num_pairs, cell_type_acc, asw = assess(mod1_embs, mod2_embs, labels_test, seed=seed)
            
            log.info(f"Metrics for replication {rep} and proportion {p}:")              
            for k,v in recall_at_k:
                print(f"k = {int(k)} | Recall@k = {v}")
            
            print(f"Number of detected true pairs = {int(num_pairs)}")
            print(f"Cell type accuracy = {cell_type_acc}")
            print(f"Average Silouette Width = {asw}")


if __name__ == "__main__":
    main()