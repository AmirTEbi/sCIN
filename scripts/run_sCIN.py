from sCIN.sCIN import train_sCIN, get_emb_sCIN
from configs import sCIN
from sCIN.utils import extract_counts, setup_logging
from sCIN.assess import assess, assess_joint_from_separate_embs
from sklearn.model_selection import train_test_split
import anndata as ad
import pandas as pd
import numpy as np
import torch 
import random
import os
import logging
import argparse



def setup_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_file", type=str)
    parser.add_argument("--atac_file", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--is_inv_metrics", action="store_true")
    parser.add_argument("--quick_test", action="store_true")
    parser.add_argument("--num_reps", type=int, default=1)  # max 10

    return parser


def main() -> None:

    parser = setup_args()
    args = parser.parse_args()

    log_save_dir = os.path.join(args.save_dir, "logs")
    setup_logging(level="info", log_dir=log_save_dir, model_name="sCIN")

    # Set seeds for replications
    seeds = [seed for seed in range(0, 100, 10)]
    if args.num_reps != 10:
        seeds = seeds[:args.num_reps]

    logging.info("Reading data files ...")
    rna_adata = ad.read_h5ad(args.rna_file)
    atac_adata = ad.read_h5ad(args.atac_file)
    rna_counts, atac_counts = extract_counts(rna_adata, atac_adata)
    labels = rna_adata.obs["cell_type_encoded"].values
    logging.info("Reading data files completed!")

    if args.quick_test :
        num_epochs = 1
        logging.info("In quick test mode, num_epochs is set to 1.")
    else:
        num_epochs = None

    res = []
    for i, seed in enumerate(seeds):

        # Control randomness
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        rep = i + 1
        logging.info(f"Replication {rep} ...")

        logging.info(f"Splitting data with seed {seed}")
        rna_train, rna_test, atac_train, atac_test, \
            labels_train, labels_test = train_test_split(rna_counts,
                                                         atac_counts,
                                                         labels,
                                                         test_size=0.3,
                                                         random_state=seed)
        logging.info(f"Training is about to start ...")
        train_dict = train_sCIN(rna_train, 
                                atac_train,
                                labels_train, 
                                settings=sCIN,
                                save_dir=args.save_dir,
                                rep=rep, 
                                is_pca=True,
                                num_epochs=num_epochs)
        
        logging.info(f"Training is about to Finished!")

        logging.info(f"Computing embeddings ...")
        rna_embs, atac_embs = get_emb_sCIN(rna_test, 
                                           atac_test,
                                           train_dict=train_dict,
                                           save_dir=args.save_dir,
                                           rep=rep)
        logging.info(f"Embeddings computed!")

        # Save outputs
        embs_save_dir = os.path.join(args.save_dir, "embs")
        os.makedirs(embs_save_dir, exist_ok=True)
        rna_embs_df = pd.DataFrame(rna_embs)
        atac_embs_df = pd.DataFrame(atac_embs)
        labels_df = pd.DataFrame(labels_test)
        rna_embs_df.to_csv(os.path.join(embs_save_dir, f"rna_embs_rep{rep}.csv"), index=False)
        atac_embs_df.to_csv(os.path.join(embs_save_dir, f"atac_embs_rep{rep}.csv"), index=False)
        labels_df.to_csv(os.path.join(embs_save_dir, f"labels_test_rep{rep}.csv"), index=False)
        logging.info(f"Embeddings and test labels saved to {embs_save_dir}.")

        # Compute metrics
        logging.info("Computing metrics on embeddings ...")
        recall_at_k_a2r, num_pairs_a2r, cell_type_acc_a2r, asw, medr_a2r = assess(atac_embs, 
                                                                                  rna_embs, 
                                                                                  labels_test, 
                                                                                  seed=seed)
        
        cell_type_acc_joint, _ = assess_joint_from_separate_embs(mod1_embs=rna_embs,
                                                              mod2_embs=atac_embs,
                                                              labels=labels_test,
                                                              seed=seed)
        if args.is_inv_metrics:
            logging.info("is_inv_metrics has been set, so metrics from RNA to ATAC will be computed ...")
            recall_at_k_r2a, num_pairs_r2a, cell_type_acc_r2a, _, medr_r2a = assess(rna_embs, 
                                                                                    atac_embs, 
                                                                                    labels_test, 
                                                                                    seed=seed)
        for k, v_a2r in recall_at_k_a2r.items():
            v_r2a = recall_at_k_r2a.get(k, 0)
            res.append({
                "Models":"sCIN",
                "Replicates":i+1,
                "k":k,
                "Recall_at_k_a2r":v_a2r,
                "Recall_at_k_r2a":v_r2a,
                "num_pairs_a2r":num_pairs_a2r,
                "num_pairs_r2a":num_pairs_r2a if num_pairs_r2a is not None else 0.0,
                "cell_type_acc_a2r":cell_type_acc_a2r,
                "cell_type_acc_r2a":cell_type_acc_r2a if cell_type_acc_r2a is not None else 0.0,
                "cell_type_ASW":asw,
                "MedR_a2r":medr_a2r,
                "MedR_r2a":medr_r2a if medr_r2a is not None else 0.0,
                "cell_type_acc_joint":cell_type_acc_joint
            })            
            
    results = pd.DataFrame(res)
    res_save_dir = os.path.join(args.save_dir, "outs")
    os.makedirs(res_save_dir, exist_ok=True)
    results.to_csv(os.path.join(res_save_dir, f"metrics_sCIN_{args.num_reps}reps.csv"), index=False)
    logging.info(f"All experiments finished and results saved to {res_save_dir}!")


if __name__ == "__main__":
    main()