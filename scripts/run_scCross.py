# from configs import scCross
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
    parser.add_argument("--num_reps", type=int)  # max 10

    return parser


def main() -> None:

    parser = setup_args()
    args = parser.parse_args()

    log_save_dir = os.path.join(args.save_dir, "logs")
    setup_logging(level="info", log_dir=log_save_dir, model_name="sCIN")

    logging.info("Reading data files ...")
    rna_adata = ad.read_h5ad(args.rna_file)
    atac_adata = ad.read_h5ad(args.atac_file)
    rna_counts, atac_counts = extract_counts(rna_adata, atac_adata)
    labels = rna_adata.obs["cell_type_encoded"].values
    logging.info("Reading data files completed!")

    # Set seeds for replications
    seeds = [seed for seed in range(0, 100, 10)]
    if args.num_reps != 10:
        seeds = seeds[:args.num_reps]
    
    res = []
    for i, seed in enumerate(seeds):
        rep = i + 1
        rna_train, rna_test, atac_train, atac_test, \
            labels_train, labels_test = train_test_split(rna_counts,
                                                         atac_counts,
                                                         labels,
                                                         test_size=0.3,
                                                         random_state=seed)
        rna_train_ad = ad.AnnData(X=rna_train)
        atac_train_ad = ad.AnnData(X=atac_train)
        rna_pca = PCA(n_components=2000)
        rna_pca.fit(rna_train)
        atac_lsi = TruncatedSVD(n_components=2000)  # Also knowns as Latent Semantic Indexing (LSI)
        atac_lsi.fit(atac_train)

        rna_train_PCs = rna_pca.transform(rna_train)
        atac_train_SVs = atac_lsi.transform(atac_train)
        rna_train_ad.uns["X_pca"] = rna_train_PCs
        atac_train_ad.uns["X_lsi"] = atac_train_SVs
        rna_train_ad.obs["cell_type"] = labels_train
        atac_train_ad.obs["cell_type"] = labels_train

        rna_test_ad = ad.AnnData(X=rna_test)
        atac_test_ad = ad.AnnData(X=atac_test)
        rna_test_PCs = rna_pca.transform(rna_test)
        atac_test_SVs = atac_lsi.transform(atac_test)
        rna_test_ad.uns["X_pca"] = rna_test_PCs
        atac_test_ad.uns["X_lsi"] = atac_test_SVs
        rna_test_ad.obs["cell_type"] = labels_test
        atac_test_ad.obs["cell_type"] = labels_test

        sccross.models.configure_dataset(rna_train_ad, 
                                         "Normal",  # Since the data has been preprocessed and normalized. 
                                         use_highly_variable=True, 
                                         use_layer = "norm_raw_counts", 
                                         use_rep="X_pca")
        
        sccross.models.configure_dataset(atac_train_ad, 
                                         "Normal", 
                                         use_highly_variable=False, 
                                         use_rep="X_lsi")
        
        sccross.models.configure_dataset(rna_test_ad, 
                                         "Normal", 
                                         use_highly_variable=True, 
                                         use_layer = "norm_raw_counts", 
                                         use_rep="X_pca")
        
        sccross.models.configure_dataset(atac_test_ad, 
                                         "Normal", 
                                         use_highly_variable=False, 
                                         use_rep="X_lsi")
        
        
        sccross.data.mnn_prior([rna_train_ad, atac_train_ad])

        # Train
        cross = sccross.models.fit_SCCROSS({"rna": rna_train_ad, 
                                            "atac": atac_train_ad}, 
                                           fit_kws={"directory": "sccross"})
        
        model_save_dir = os.path.join(args.save_dir, "models")
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Get embeddings
        rna_embs = cross.encode_data("rna", rna_test_ad)
        atac_embs = cross.encode_data("atac", atac_test_ad)

        embs_save_dir = os.path.join(args.save_dir, "embs")
        os.makedirs(embs_save_dir, exist_ok=True)
        rna_embs_df = pd.DataFrame(rna_embs)
        atac_embs_df = pd.DataFrame(atac_embs)
        rna_embs_df.to_csv(os.path.join(embs_save_dir, f"rna_embs_rep{rep}.csv"), index=False)
        atac_embs_df.to_csv(os.path.join(embs_save_dir, f"atac_embs_rep{rep}.csv"), index=False)

        # Evals
        recall_at_k_a2r, num_pairs_a2r, cell_type_acc_a2r, asw, medr_a2r = assess(atac_embs,
                                                                                  rna_embs,
                                                                                  labels,
                                                                                  seed=seed)
        if args.is_inv_metrics:
            recall_at_k_r2a, num_pairs_r2a, cell_type_acc_r2a, _, medr_r2a = assess(rna_embs,
                                                                                    atac_embs,
                                                                                    labels,
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
    results.to_csv(os.path.join(res_save_dir, f"metrics_scCross_{args.num_reps}reps.csv"), index=False)
    logging.info(f"scCross results saved to {res_save_dir}.")


if __name__ == "__main__":
    main()