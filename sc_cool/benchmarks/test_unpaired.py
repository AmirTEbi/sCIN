import numpy as np
import pandas as pd
import scanpy as sc
from sc_cool.utils.utils import (split_full_data,
                                   split_partial_data,
                                   extract_counts, 
                                   get_func_name_unpaired, 
                                   read_config, 
                                   load_data, 
                                   compute_KL_loss, 
                                   guassian_kernel, 
                                   mmd_rbf, accuracy, 
                                   dis_accuracy,    
                                   train_autoencoders, 
                                   train_classifier, 
                                   make_plots,
                                   select_unpaired_cells_by_type)

from sc_cool.models.sc_cool import (Mod1Encoder, 
                                      Mod2Encoder, 
                                      scCOOL, train_sccool_unpaired, 
                                      get_emb_sccool)
from sc_cool.models.ConAAE.con_aae import (setup_args, train_con_unpaired, get_emb_con)

from sc_cool.models.AE import (Mod1Encoder, 
                               Mod2Encoder, 
                               Mod1Decoder, 
                               Mod2Decoder,
                               SimpleAutoEncoder, 
                               train_ae_unpaired, 
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
import time
import argparse
import os


def main():

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

        train_func_name, get_emb_func_name = get_func_name_unpaired(model_name)
        train_func = globals().get(train_func_name)
        get_emb_func = globals().get(get_emb_func_name)

        print(f"Experiment for model {model_name} has been started!")


        #############
        if args.quick_test:
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

                rna_train, rna_test, atac_train, atac_test, \
                labels_train, labels_test = split_partial_data(mod1, mod2, proportion=settings["PROPORTION"], seed=seed)
                
            else:
                
                rna_train, rna_test, atac_train, atac_test, \
                    labels_train, labels_test = split_full_data(mod1, mod2, seed=seed)


            print(rna_train.shape)
            print(atac_train.shape)
            print(rna_test.shape)
            print(atac_test.shape)
            
            print("Data splitted!")
            print(f"Training has been started!")
            TrainTime0 = time.time()

            obj_list = train_func(rna_train, atac_train, labels_train, epochs=epochs, 
                                  settings=settings, device=device, seed=seed)
            
            TrainTime1 = time.time()
            print(f"Total training time for model {model_name}: {(TrainTime1 - TrainTime0)/60} minutes.")
            print("Training has been ended!")

            # Get embeddings
            rna_emb, atac_emb = get_emb_func(rna_test, atac_test, labels_test, obj_list, save_dir=config["SAVE_DIRS"][model_name], seed=seed, device=device)
            print(f"Shape of RNA embeddings: {rna_emb.shape}")
            print(f"Shape of ATAC embeddings: {atac_emb.shape}")
            print(f"Embeddings for model {model_name} were generated!")

            # Assessment
            print("Assessment has been started!")
            recall_at_k, num_pairs, class_lbl_acc, asw = assess(rna_emb, atac_emb, labels_test, n_pc=20,
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