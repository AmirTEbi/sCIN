from sCIN.sCIN_random_pairs_unpaired import train_sCIN_ablated_unpaired, get_embs_sCIN_ablated_unpaired
from configs import sCIN
from sCIN.utils import extract_counts, setup_logging
from sCIN.assess import assess_unpaired
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
    parser.add_argument("--num_reps", type=int)  # max 10

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
    rna_labels = rna_adata.obs["cell_type"].values
    atac_labels = atac_adata.obs["cell_type"].values
    logging.info("Reading data files completed!")

    if args.quick_test :
        num_epochs = 1
        logging.info("In quick test mode, num_epochs is set to 1.")
    else:
        num_epochs = None

    res = []
    for i, seed in enumerate(seeds):
        rep = i + 1
        logging.info(f"Replication {rep} ...")

        rna_train, rna_test, rna_lbls_train, rna_lbls_test = train_test_split(rna_counts,
                                                                              rna_labels,
                                                                              test_size=0.3,
                                                                              random_state=seed)
        
        atac_train, atac_test, atac_lbls_train, atac_lbls_test = train_test_split(atac_counts,
                                                                                  atac_labels,
                                                                                  test_size=0.3,
                                                                                  random_state=seed)
        logging.info("Training is about to start ...")
        train_dict = train_sCIN_ablated_unpaired(mod1_train=rna_train,
                                                 mod2_train=atac_train,
                                                 mod1_labels_train=rna_lbls_train,
                                                 mod2_labels_train=atac_lbls_train,
                                                 settings=sCIN,
                                                 save_dir=args.save_dir,
                                                 is_pca=True,
                                                 rep=rep,
                                                 num_epochs=num_epochs)
        logging.info("Training completed.") 

        logging.info(f"Computing embeddings ...")
        rna_embs, atac_embs = get_embs_sCIN_ablated_unpaired(rna_test, 
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
        rna_labels_df = pd.DataFrame(rna_lbls_test)
        atac_labels_df = pd.DataFrame(atac_lbls_test)
        rna_embs_df.to_csv(os.path.join(embs_save_dir, f"rna_embs_rep{rep}.csv"), index=False)
        atac_embs_df.to_csv(os.path.join(embs_save_dir, f"atac_embs_rep{rep}.csv"), index=False)
        rna_labels_df.to_csv(os.path.join(embs_save_dir, f"RNA_labels_test_rep{rep}.csv"), index=False)
        atac_labels_df.to_csv(os.path.join(embs_save_dir, f"ATAC_labels_test_rep{rep}.csv"), index=False)
        logging.info(f"Embeddings and test labels saved to {embs_save_dir}.")

        # Compute metrics
        logging.info("Computing metrics on embeddings ...")
        ct_at_k_a2r, asw, GC_uni, GC_joint, GC_union = assess_unpaired(mod1_embs=atac_embs,
                                                                       mod2_embs=rna_embs,
                                                                       mod1_labels=atac_lbls_test,
                                                                       mod2_labels=rna_lbls_test,
                                                                       seed=seed)
        if args.is_inv_metrics:
            ct_at_k_r2a, _, _, _, _ = assess_unpaired(mod1_embs=rna_embs,
                                                      mod2_embs=atac_embs,
                                                      mod1_labels=rna_lbls_test,
                                                      mod2_labels=atac_lbls_test,
                                                      seed=seed)
        
        for k, v_a2r in ct_at_k_a2r.items():
            v_r2a = ct_at_k_r2a.get(k, 0)
            res.append({
                "Models":"sCIN-ablated",
                "Replicates":i+1,
                "k":k,
                "cell_type_at_k_a2r":v_a2r,
                "cell_type_at_k_r2a":v_r2a if v_r2a is not None else 0.0,
                "cell_type_ASW":asw,
                "GC_uni":GC_uni,
                "GC_joint":GC_joint,
                "GC_union":GC_union
            })
            
   
    results = pd.DataFrame(res)
    res_save_dir = os.path.join(args.save_dir, "outs")
    os.makedirs(res_save_dir, exist_ok=True)
    results.to_csv(os.path.join(res_save_dir, f"metrics_sCIN_random_pairs_unpaired{args.num_reps}reps.csv"), index=False)
    logging.info(f"All experiments finished and results saved to {res_save_dir}!")


if __name__ == "__main__":
    main()