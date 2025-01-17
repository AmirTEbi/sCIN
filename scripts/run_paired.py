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
from sCIN.models.AE import (Mod1Encoder, 
                               Mod2Encoder, 
                               Mod1Decoder, 
                               Mod2Decoder,
                               SimpleAutoEncoder, 
                               train_ae, 
                               get_emb_ae) 
from sCIN.models.ConAAE.con_aae import (setup_args, train_con, get_emb_con)
from sCIN.models.harmony import (train_hm, get_emb_hm)
from sCIN.models.mofa import (prepare_data_mofa, 
                                extract_embs, 
                                train_mofa, 
                                get_mofa_emb)
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

MODELS = ("sCIN", "Con-AAE", "MOFA", "Harmony", "AE")
DATA = {"PBMC":("data/10x/10x-Multiome-Pbmc10k-RNA.h5ad", "data/10x/10x-Multiome-Pbmc10k-ATAC.h5ad"),
        "SHARE": ("data/share/Ma-2020-RNA.h5ad", "data/share/Ma-2020-ATAC.h5ad"),
        "CITE":("data/cite/rna.h5ad", "data/cite/adt.h5ad")}

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str)
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--data", type=str)
    parser.add_argument("--seed_range", type=int, default=100)
    parser.add_argument("--outfile", type=str)
    parser.add_argument("--quick_test", action="store_true")
    args = parser.parse_args()

    config = read_config(args.cfg_path)
    SETTINGS = config["SETTINGS"]
    BASE_SAVE_DIR = f"results/{args.data}/{args.model}/V1"
    #seeds = list(np.arange(0, args.seed_range, 10))
    seeds = [90]
    epochs = 1 if args.quick_test else SETTINGS.get("EPOCHS")

    mod1_path, mod2_path = DATA[args.data]
    mod1 = ad.read_h5ad(mod1_path)
    mod2 = ad.read_h5ad(mod2_path)
    print("Data loaded successfully.")

    res = pd.DataFrame(columns=["Models", "Replicates", "k", 
                                "Recall_at_k", "Recall_at_k_2to1", 
                                "num_pairs", "num_pairs_2to1", "cell_type_acc",
                                "cell_type_acc_2to1", "cell_type_ASW", "rep_time(min)"])
    
    if args.model == "all":
        models = MODELS
    else:
        models = [args.model]
    
    start_pipeline_time = time.time()
    train_times = {model:0.0 for model in models}
    rep_times = {model:0.0 for model in models}
    for model in models:
        train_func_name, get_emb_func_name = get_func_name(model)
        train_func = globals().get(train_func_name)
        get_emb_func = globals().get(get_emb_func_name)

        start_rep_time = time.time()
        for seed in seeds:
            print(f"Seed is {seed}")
            save_dir_seed = os.path.join(BASE_SAVE_DIR, f"rep{int(seed/10 + 1)}")
            os.makedirs(save_dir_seed, exist_ok=True)

            mod1_train, mod1_test, mod2_train, mod2_test, \
                labels_train, labels_test = split_full_data(mod1, mod2, seed=seed)
            print("Data splitted.")

            start_train_time = time.time()
            print("Training has been started ...")
            train_dict = train_func(mod1_train, mod2_train, labels_train,
                                    epochs=epochs, settings=SETTINGS,
                                    seed=seed, is_pca=True, save_dir=save_dir_seed)
            end_train_time = time.time()
            train_time = (end_train_time - start_train_time)/60
            train_times[model] += train_time

            print("Evaluation has been started ...")
            mod1_embs, mod2_embs = get_emb_func(mod1_test, mod2_test, labels_test, 
                                                train_dict, save_dir_seed,
                                                seed=seed, is_pca=True)
            recall_at_k, num_pairs, cell_type_acc, asw = assess(mod1_embs, mod2_embs, 
                                                                labels_test, seed=seed)
            recall_at_k_2to1, num_pairs_2to1, cell_type_acc_2to1 = compute_metrics(mod1_embs,
                                                                                   mod2_embs, 
                                                                                   labels_test)
            
            end_rep_time = time.time()
            rep_time = (end_rep_time - start_rep_time)/60
            rep_times[model] += rep_time
            
            for k, v1 in recall_at_k.items():
                v2 = recall_at_k_2to1.get(k, 0)
                new_row = pd.DataFrame({
                    "Models": [model],
                    "Replicates": [seeds.index(seed) + 1],
                    "k": [k],
                    "Recall_at_k": [v1],
                    "Recall_at_k_2to1": [v2],
                    "num_pairs": [num_pairs],
                    "num_pairs_2to1": [num_pairs_2to1],
                    "cell_type_acc": [cell_type_acc],
                    "cell_type_acc_2to1": [cell_type_acc_2to1],
                    "cell_type_ASW": [asw],
                    "rep_time(min)":[rep_time]})
                
                res = pd.concat([res, new_row], ignore_index=True)
                res.to_csv(os.path.join(save_dir_seed, 
                                        f"results_rep_{seeds.index(seed) + 1}.csv"), 
                                        index=False)

            print(f"Results saved for replication {seeds.index(seed) + 1}.")
        
        res.to_csv(os.path.join(BASE_SAVE_DIR, f"{args.outfile}.csv"), index=False)
    
    for model, t in train_times.items():
        print(f"Training {model} takes {t/10} mins in average")
        rep_time = rep_times.get(model)
        print(f"A replication for model takes {rep_time/10} in average.")

    end_pipeline_time = time.time()
    pipeline_time = (end_pipeline_time - start_pipeline_time)/60
    print(f"Total pipeline takes {pipeline_time} mins.")
    print("Finished.")


if __name__ == "__main__":
    main()