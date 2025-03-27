from scButterfly.butterfly import Butterfly
from ..sCIN.assess import assess
import anndata as ad 
import torch
import random
import argparse
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
    parser.add_argument("--num_reps", type=int)  # max 10
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
        for i in range(butterfly.ATAC_data.var.chrom):
            temp = butterfly.ATAC_data_p.var.chrom[i]
            if temp[0 : 3] == 'chr':
                if not temp == last_one:
                    chrom_list.append(1)
                    last_one = temp
                else:
                    chrom_list[-1] += 1
            else:
                chrom_list[-1] += 1
        
        print(f"Total number of peaks: {sum(chrom_list)}")

        # Construct the model
        butterfly.construct_model(chrom_list=chrom_list)

        # Train and save the model
        butterfly.train_model(seed=seed)
        model_save_dir = os.path.join(args.save_dir, "models")
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        torch.save(butterfly.state_dict(), os.path.join(model_save_dir, f"scButterfly_rep{i}.pt"))

        # Get Predictions
        A2R_predict, R2A_predict = butterfly.test_model()
        print(f"Shape of the ATAC to RNA pred: {A2R_predict.shape}")
        print(f"Shape of the RNA to ATAC pred: {R2A_predict.shape}")

        # Evaluations
        recall_at_k, num_pairs, cell_type_acc, asw = assess(A2R_predict,
                                                            R2A_predict,
                                                            )









if __name__ == "__main__":
    main()