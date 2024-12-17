"""
This is a script for running unpaired data setting with Optimal Transport.
Author: Amir Ebrahimi
Date: 2024-12-17
"""

import numpy as np
import pandas as pd
import scanpy as sc
from sc_cool.utils.utils import (split_full_data,
                                   extract_counts, 
                                   get_func_name, 
                                   read_config, 
                                   load_data, 
                                   compute_KL_loss, 
                                   guassian_kernel, 
                                   mmd_rbf, accuracy, 
                                   dis_accuracy,    
                                   train_autoencoders, 
                                   train_classifier, 
                                   make_plots,
                                   remove_rows,
                                   make_extreme_unpaired)

from sc_cool.models.sc_cool import (RNAEncoder, 
                                      ATACEncoder, 
                                      scCOOL, train_sccool, 
                                      get_emb_sccool)
from sc_cool.models.ConAAE.con_aae import (setup_args, train_con, get_emb_con)

from sc_cool.models.AE import (Mod1Encoder, 
                               Mod2Encoder, 
                               Mod1Decoder, 
                               Mod2Decoder,
                               SimpleAutoEncoder, 
                               train_ae, 
                               get_emb_ae)

from sc_cool.benchmarks.assess import (ct_recall, assess)
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import ot
import time
import argparse
import os


def main():

    ############################################### GLOBAL SETUP ############################################


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read data path from the config file
    parser = argparse.ArgumentParser(description='Get config')
    parser.add_argument('--config_path', type=str, help='config file')
    parser.add_argument('--seed_range', type=int, help='Range of the random seeds', default=100)
    parser.add_argument("--quick_test", action='store_true', help="Whether to run training for one epoch for debuging")
    args = parser.parse_args()

    config = read_config(args.config_path)

    # Load data
    mod1, mod2 = load_data(config["DATA_DIR"])
    print("Data has been loaded!")

    seeds = list(np.arange(0, args.seed_range, 10))
    print(f"Seeds are: {seeds}")
    
    results_df = pd.DataFrame(columns=["Models", "Replicates", "k", "Recall_at_k", "num_pairs", 
                                       "cell_type_acc", "cell_type_ASW"])

    # Determine the name of the model to train
    for i, name in enumerate(config["MODEL_NAMES"]):

        StartTime = time.time()
        model_name = name
        settings = config["SETTINGS"][model_name]

        train_func_name, get_emb_func_name = get_func_name(model_name)
        train_func = globals().get(train_func_name)
        get_emb_func = globals().get(get_emb_func_name)

        print(f"Experiment for model {model_name} has been started!")


        ################################################## DEBUG ###############################################
        if args.quick_test:
            epochs = 1
        else:
            epochs = settings["EPOCHS"]
        ############################################### REPLICATIONS ###########################################

        for seed in seeds:
            print(f"Seed {seed} has been started!")

            # To save the results for the current replication
            results_rep_df = pd.DataFrame(columns=["Models", "Replicates", "k", "Recall_at_k", "num_pairs", 
                                                   "cell_type_acc", "cell_type_ASW"])  
                          
        ############################################### DATA SPLITTING ##########################################
        
            mod1_train, mod1_test, mod2_train, mod2_test, \
                    labels_train, labels_test = split_full_data(mod1, mod2, seed=seed)
            

        ############################################### UNPAIRING ################################################

            # 1) Unpairing
            mod1_train_unpaired, mod2_train_unpaired = make_extreme_unpaired(mod1_train,
                                                                             mod2_train,
                                                                             labels_train,
                                                                             seed=seed)
            
            print(f"Shape of the Mod1 unapired dataset: {mod1_train_unpaired.shape}")
            print(f"Shape of the Mod2 unapired dataset: {mod2_train_unpaired.shape}")

            # 2) OT (Sinkhorn algorithm)
            M = ot.dist(mod1_train_unpaired, mod2_train_unpaired, metric='euclidean') # Distance between two distributions

            regs = [0.01, 0.05, 0.1, 0.5]
            regs_dict = {}
            for reg in regs:
                G = ot.sinkhorn(  # Compute transportation matrix
                    torch.ones(mod1_train_unpaired.shape[0]) / mod1_train_unpaired.shape[0],
                    torch.ones(mod2_train_unpaired.shape[0]) / mod2_train_unpaired.shape[0],
                    torch.tensor(M), reg=reg
                )
                cost = torch.sum(G * torch.tensor(M)).item() # Transportation cost
                regs_dict[reg] = {"G": G, "cost": cost}
            
            best_reg = min(regs_dict, key=lambda r: regs_dict[r]['cost'])
            best_G = regs_dict[best_reg]['G']
            print(f"Best regularization value for Sinkhorn algorithm is: {best_reg}")

            mod2_transported = torch.mm(best_G, torch.tensor(mod2_train_unpaired)).numpy() * mod1_train_unpaired.shape[0]

            # Save the best transporation matrix for this replication
            out = os.path.join(config["SAVE_DIRS"][model_name], f"best_G_rep{seed}.npy")
            np.save(out, best_G.cpu().numpy())

            ############################################ TRAINING ####################################################
                    
            
            TrainTime0 = time.time()

            obj_list = train_func(mod1_train_unpaired, mod2_transported, labels_train, epochs=epochs, 
                                  settings=settings, device=device, seed=seed)
            
            TrainTime1 = time.time()
            print(f"Total training time for model {model_name}: {(TrainTime1 - TrainTime0)/60} minutes.")
            print("Training has been ended!")

            ############################################ EVALUATIONS ####################################################

            # Get embeddings
            mod1_embs, mod2_embs = get_emb_func(mod1_test, mod2_test, labels_test, obj_list, save_dir=config["SAVE_DIRS"][model_name], seed=seed, device=device)
            print(f"Shape of Mod1 embeddings: {mod1_embs.shape}")
            print(f"Shape of Mod2 embeddings: {mod2_embs.shape}")
            print(f"Embeddings for model {model_name} were generated!")

            # Assessment
            print("Assessment has been started!")
            recall_at_k, num_pairs, class_lbl_acc, asw = assess(mod1_embs, mod2_embs, labels_test, n_pc=20,
                                            save_path=config["SAVE_DIRS"][model_name], seed=seed)
            print(type(recall_at_k))
            print("Assessment completed!")

            for k,v in recall_at_k.items():
                new_row = pd.DataFrame({
                    "Models": [model_name],
                    "Replicates": [seeds.index(seed) + 1],
                    "k": k,
                    "Recall_at_k": [v],
                    "num_pairs": [num_pairs],
                    "cell_type_acc":[class_lbl_acc],
                    "cell_type_ASW":asw
                })


                results_rep_df = pd.concat([results_rep_df, new_row], ignore_index=True)
                results_rep_df.to_csv(config["SAVE_DIRS"][model_name] + f"/results_rep_{seeds.index(seed) + 1}.csv", index=False)
                results_df = pd.concat([results_df, new_row], ignore_index=True)


            EndTime = time.time()
            print(f"Total pipeline runtime for the model {model_name} is: {(EndTime - StartTime)/60} minutes.")
            print(f"Experiment for model {model_name} has been finished!")

    experiment = config["EXPERIMENT_NAME"]
    results_df.to_csv(config["SAVE_DIRS"]["EXP"] + f"/results_{experiment}.csv", index=False)
    make_plots(results_df, config["SAVE_DIRS"]["EXP"])

    print("All experiments has been finished!")

if __name__ == "__main__":
    main()