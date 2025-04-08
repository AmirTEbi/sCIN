from scButterfly.butterfly import Butterfly
from ..sCIN.assess import assess
import anndata as ad 
import pandas as pd
import torch
import random
import argparse
import logging
import os


def five_fold_split_dataset(
    RNA_data, 
    ATAC_data, 
    seed = 19193
):
    """
    Adapted from https://github.com/BioX-NKU/scButterfly. 
    """
    
    # if not seed is None:
    #     setup_seed(seed)
    random.seed(seed)
    temp = [i for i in range(len(RNA_data.obs_names))]
    random.shuffle(temp)
    
    id_list = []
    
    test_count = int(0.2 * len(temp))
    validation_count = int(0.16 * len(temp))
    
    for i in range(5):
        test_id = temp[: test_count]
        validation_id = temp[test_count: test_count + validation_count]
        train_id = temp[test_count + validation_count:]
        temp.extend(test_id)
        temp = temp[test_count: ]

        id_list.append([train_id, validation_id, test_id])
    
    return id_list


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_file", type=str)
    parser.add_argument("--atac_file", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--num_reps", type=int, choices=range(1, 11), help="Number of replication(1-10)")
    parser.add_argument("--is_inv_metrics", action="store_true")
    args = parser.parse_args()

    # Read data
    RNA_data = ad.read_h5ad(args.rna_file)
    ATAC_data = ad.read_h5ad(args.atac_file)

    # Set seeds
    seeds = [seed for seed in range(0, 100, 10)]
    if args.num_reps != 10:
        seeds = seeds[:args.num_reps]
    
    res = []  # To save the results
    for i, seed in enumerate(seeds):

        # Initialize the model object
        butterfly = Butterfly()
        # Split data to train and test
        id_list = five_fold_split_dataset(RNA_data, ATAC_data, seed=seed)
        train_id, _, test_id = id_list[4]

        # Load data
        butterfly.load_data(RNA_data, ATAC_data, train_id, test_id)
        butterfly.is_processed = True

        # Preprocessing data
        # butterfly.data_preprocessing()

        # Make a chrom list. For more information, please see: https://scbutterfly.readthedocs.io/en/latest/Tutorial/RNA_ATAC_paired_prediction/RNA_ATAC_paired_scButterfly-B.html
        chrom_list = []
        last_one = ""
        for i in range(len(butterfly.ATAC_data.var.chrom)):
            temp = butterfly.ATAC_data_p.var.chrom[i]
            if temp[0 : 3] == 'chr':
                if not temp == last_one:
                    chrom_list.append(1)
                    last_one = temp
                else:
                    chrom_list[-1] += 1
            else:
                chrom_list[-1] += 1
        
        logging.info(f"Total number of peaks: {sum(chrom_list)}")

        # Construct the model
        butterfly.construct_model(chrom_list=chrom_list)

        # Train and save the model
        butterfly.train_model(seed=seed)
        model_save_dir = os.path.join(args.save_dir, "models")
        os.makedirs(model_save_dir, exist_ok=True)
        torch.save(butterfly.state_dict(), os.path.join(model_save_dir, f"scButterfly_rep{i}.pt"))

        # Get Predictions
        A2R_predict, R2A_predict = butterfly.test_model()
        print(f"Shape of the ATAC to RNA pred: {A2R_predict.shape}")
        print(f"Shape of the RNA to ATAC pred: {R2A_predict.shape}")
        labels = RNA_data.obs["cell_type_encoded"].values
        labels_test = labels[test_id]

        rna_embs_df = pd.DataFrame(A2R_predict)
        atac_embs_df = pd.DataFrame(R2A_predict)
        labels_test_df = pd.DataFrame(labels_test)
        embs_save_dir = os.path.join(args.save_dir, "embs")
        os.makedirs(embs_save_dir, exist_ok=True)
        rna_embs_df.to_csv(os.path.join(embs_save_dir, f"rna_embs_rep{i}.csv"), index=False)
        atac_embs_df.to_csv(os.path.join(embs_save_dir, f"atac_embs_rep{i}.csv"), index=False)
        labels_test_df.to_csv(os.path.join(embs_save_dir, f"labels_test_rep{i}.csv"), index=False)


        # Evaluations
        recall_at_k_a2r, num_pairs_a2r, cell_type_acc_a2r, asw, medr_a2r = assess(R2A_predict,
                                                                                  A2R_predict,
                                                                                  labels_test,
                                                                                  seed=seed)
        if args.is_inv_metrics:
            recall_at_k_r2a, num_pairs_r2a, cell_type_acc_r2a, _, medr_r2a = assess(A2R_predict,
                                                                                    R2A_predict,
                                                                                    labels_test,
                                                                                    seed=seed)
        for k, v_a2r in recall_at_k_a2r.items():
            v_r2a = recall_at_k_r2a.get(k, 0)
            res.append({
                "Models":"scButterfly",
                "Replicates":i+1,
                "k":k,
                "Recall_at_k_a2r":v_a2r,
                "Recall_at_k_r2a":v_r2a if v_r2a is not None else 0.0,
                "num_pairs_a2r":num_pairs_a2r,
                "num_pairs_r2a":num_pairs_r2a if num_pairs_r2a is not None else 0.0,
                "cell_type_acc_a2r":cell_type_acc_a2r,
                "cell_type_acc_r2a":cell_type_acc_r2a if cell_type_acc_r2a is not None else 0.0,
                "cell_type_ASW":asw,
                "MedR_a2r":medr_a2r,
                "MedR_r2a":medr_r2a if medr_r2a is not None else 0.0
            })
    
    results = pd.DataFrame(res)
    res_save_dir = os.path.join(args.save_dir, "outs")
    os.makedirs(res_save_dir, exist_ok=True)
    results.to_csv(os.path.join(res_save_dir, f"metrics_scButterfly_{args.num_reps}reps.csv"), index=False)
    logging.info(f"scButterfly results saved to {res_save_dir}.")


if __name__ == "__main__":
    main()