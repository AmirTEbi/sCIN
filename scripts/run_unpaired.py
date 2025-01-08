"""
Script for reproducing results for unpaired settings.

To run:
cd sc-cool
sc-cool > python scripts/run_unpaired.py --model_cfg "..." --data "..." --seed_range  --outfile "..."
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sc_cool.utils.utils import(split_full_data,
                                   extract_counts, 
                                   read_config, 
                                   make_plots,
                                   make_unpaired,
                                   make_unpaired_random)
from sc_cool.models.sc_cool import (Mod1Encoder, 
                                    Mod2Encoder, 
                                    scCOOL, 
                                    train_sCIN_unpaired, 
                                    get_emb_sCIN, 
                                    pca_with_nans)
from sc_cool.benchmarks.assess import (compute_metrics, assess)
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

def main() -> None:

    parser = argparse.ArgumentParser(description='Get config')
    parser.add_argument("--model_cfg", type=str)
    parser.add_argument("--data", type=str)
    parser.add_argument("--seed_range", type=int, default=100)
    parser.add_argument("--outfile", type=str)
    parser.add_argument("--quick_test", action='store_true')
    args = parser.parse_args()

    config = read_config(args.model_cfg)

    ############### Main Setup ###################
    SETTINGS = config["SETTINGS"]
    BASE_SAVE_DIR = f"results/{args.data}/main_unpaired"
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)

    
    seeds = list(np.arange(0, args.seed_range, 10))
    print(f"Seeds are: {seeds}")

    p_list = ("Random", 0.01, 0.05, 0.1, 0.2, 0.5)

    epochs = 1 if args.quick_test else SETTINGS.get("EPOCHS", 150)
    print(f"Epochs: {epochs}")
    
    DATA_PATHS = {
        "PBMC": ("data/10x/10x-Multiome-Pbmc10k-RNA.h5ad", "data/10x/10x-Multiome-Pbmc10k-ATAC.h5ad"),
        "SHARE": ("data/share/Ma-2020-RNA.h5ad", "data/share/Ma-2020-ATAC.h5ad"),
        "CITE": ("data/cite/rna.h5ad", "data/cite/adt.h5ad")}
    
    mod1_path, mod2_path = DATA_PATHS[args.data]
    mod1 = ad.read_h5ad(mod1_path)
    mod2 = ad.read_h5ad(mod2_path)
    print("Data loaded successfully.")

    results_df = pd.DataFrame(columns=["Models", "Replicates", "k", 
                                        "Mod2_prop", "Recall_at_k", 
                                        "Recall_at_k_2to1", "num_pairs", 
                                        "num_pairs_2to1", "cell_type_acc",
                                        "cell_type_acc_2to1", "cell_type_ASW"])
    ###############################################

    for seed in seeds:
        print(f"Current seed: {seed}")

        save_dir_seed = os.path.join(BASE_SAVE_DIR, f"rep{seed}")
        os.makedirs(save_dir_seed, exist_ok=True)
        
        RepTime0 = time.time()
        
        mod1_train, mod1_test, mod2_train, mod2_test, \
            labels_train, labels_test = split_full_data(mod1, mod2, seed=seed)
        print("Data splitted!")

        for i in p_list:
            print(f"Current p: {i}")
            model_name = f"sCIN_{i}" if i != "Random" else "sCIN_Random"
            save_dir_p = os.path.join(save_dir_seed, 
                                      f"p{int(i * 100) if i != 'Random' else 'Random'}")
            os.makedirs(save_dir_p, exist_ok=True)

            if i == "Random":
                mod1_train_unp, mod1_lbls_unp, mod2_train_unp, mod2_lbls_unp = \
                    make_unpaired_random(mod1_train, mod2_train, labels_train, seed, num_mod2_ct=5)
            else:
                mod1_train_unp, mod1_lbls_unp, mod2_train_unp, mod2_lbls_unp = \
                    make_unpaired(mod1_train, mod2_train, labels_train, seed, p=i)
            
            TrainTime0 = time.time()

            train_dict = train_sCIN_unpaired(mod1_train_unp, mod2_train_unp, 
                                             [mod1_lbls_unp, mod2_lbls_unp], 
                                             epochs=epochs, settings=SETTINGS, 
                                             seed=seed, save_dir=save_dir_p, 
                                             is_pca=True)
            TrainTime1 = time.time()
            train_p_time = TrainTime1 - TrainTime0
            print(f"Total training time for rep {seeds.index(seed)+1} | p = {i}: \
                  {(train_p_time)/60} minutes.")
            
            mod1_embs, mod2_embs = get_emb_sCIN(mod1_test, mod2_test, 
                                                labels_test, train_dict, 
                                                save_dir=save_dir_p, seed=seed,
                                                is_pca=True)
            
            recall_at_k, num_pairs, cell_type_acc, asw = assess(mod1_embs, mod2_embs, 
                                                                labels_test, n_pc=20, 
                                                                save_dir=save_dir_p, 
                                                                seed=seed)
            recall_at_k_2to1, num_pairs_2to1, cell_type_acc_2to1 = compute_metrics(mod2_embs,
                                                                                   mod1_embs, 
                                                                                   labels_test)


            for k, v1 in recall_at_k.items():
                v2 = recall_at_k_2to1.get(k, 0)
                new_row = pd.DataFrame({
                    "Models": [model_name],
                    "Replicates": [seeds.index(seed) + 1],
                    "k": [k],
                    "Mod2_prop": [i],
                    "Recall_at_k": [v1],
                    "Recall_at_k_2to1": [v2],
                    "num_pairs": [num_pairs],
                    "num_pairs_2to1": [num_pairs_2to1],
                    "cell_type_acc": [cell_type_acc],
                    "cell_type_acc_2to1": [cell_type_acc_2to1],
                    "cell_type_ASW": [asw]})

                results_df = pd.concat([results_df, new_row], ignore_index=True)

            results_df.to_csv(os.path.join(save_dir_p, 
                                            f"results_rep_{seeds.index(seed) + 1}.csv"), 
                                            index=False)
            print(f"Results saved for seed {seed} and prop {i}.")

        RepTime1 = time.time()
        rep_time = RepTime1 - RepTime0
        print(f"Total replication time: {rep_time/60} minutes.")
    
    results_df.to_csv(os.path.join(BASE_SAVE_DIR, f"{args.outfile}.csv"), index=False)
    make_plots(results_df, BASE_SAVE_DIR)
    print("Finished.")


if __name__ == "__main__":
    main()