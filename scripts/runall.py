"""
This is a driver script that runs all necessary experiments.
"""

# Imports
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sc_cool.utils.utils import (split_full_data,
                                   split_partial_data,
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
                                   make_plots)
from sc_cool.models.sc_cool import (RNAEncoder, 
                                      ATACEncoder, 
                                      scCOOL, train_sccool, 
                                      get_emb_sccool)
from sc_cool.models.AE import (RNAEncoderAE, 
                               ATACEncoderAE, 
                               RNADecoder, 
                               ATACDecoder,
                               SimpleAutoEncoder, 
                               train_ae, 
                               get_emb_ae) 
from sc_cool.models.ConAAE.con_aae import (setup_args, train_con, get_emb_con)
from sc_cool.models.harmony import (train_hm, get_emb_hm)
from sc_cool.models.mofa import (prepare_data_mofa, 
                                   extract_embs, 
                                   train_mofa, 
                                   get_mofa_emb)
from sc_cool.benchmarks.assess import (ct_recall, assess)
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


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read data path from the config file
    parser = argparse.ArgumentParser(description='Get config')
    parser.add_argument('--config_path', type=str, help='config file')
    parser.add_argument('--seed_range', type=int, help='Range of the random seeds for replication', default=100)
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

        #############
        QUICK_TEST = False
        if QUICK_TEST:
            epochs = 1
        else:
            epochs = settings["EPOCHS"]
        #############
            
        for seed in seeds:
            print(f"Seed {seed} has been started!")

            # To save the results for the current replication
            results_rep_df = pd.DataFrame(columns=["Models", "Replicates", "k", "Recall_at_k", "num_pairs", 
                                                   "cell_type_acc", "cell_type_ASW"])                

            # Split data into train and test
            if settings["IS_PARTIAL"] == "True":

                mod1_train, mod1_test, mod2_train, mod2_test, \
                labels_train, labels_test = split_partial_data(mod1, mod2, proportion=settings["PROPORTION"], seed=seed)
                
            else:
                
                mod1_train, mod1_test, mod2_train, mod2_test, \
                    labels_train, labels_test = split_full_data(mod1, mod2, seed=seed)


            print(mod1_train.shape)
            print(mod2_train.shape)
            print(mod1_test.shape)
            print(mod2_test.shape)
            
            print("Data splitted!")
            print("****************************")
            print(f"Training has been started!")
            TrainTime0 = time.time()

            obj_list = train_func(mod1_train, mod2_train, labels_train, epochs=epochs,
                settings=settings, device=device)
            
            TrainTime1 = time.time()
            print(f"Total training time for model {model_name}: {(TrainTime1 - TrainTime0)/60} minutes.")
            print("Training has been ended!")
            print("****************************")


            # Get embeddings
            mod1_emb, mod2_emb = get_emb_func(mod1_test, mod2_test, labels_test, obj_list, save_dir=config["SAVE_DIRS"][model_name], seed=seed, device=device)
            print(f"Shape of RNA embeddings: {mod1_emb.shape}")
            print(f"Shape of ATAC embeddings: {mod2_emb.shape}")
            print(f"Embeddings for model {model_name} were generated!")
            print("****************************")

            # Assessment
            print("Assessment has been started!")
            recall_at_k, num_pairs, class_lbl_acc, asw = assess(mod1_emb, mod2_emb, labels_test, n_pc=20,
                                            save_path=config["SAVE_DIRS"][model_name], seed=seed)
            print(type(recall_at_k))
            print("Assessment completed!")
            print("****************************")

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
            print("****************************")

    experiment = config["EXPERIMENT_NAME"]
    results_df.to_csv(config["SAVE_DIRS"]["EXP"] + f"/results_{experiment}.csv", index=False)
    make_plots(results_df, config["SAVE_DIRS"]["EXP"])

    print("All experiments has been finished!")

if __name__ == "__main__":
    main()