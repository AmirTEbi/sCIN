from sCIN.sCIN import train_sCIN, get_emb_sCIN
from configs import sCIN
from sCIN.utils import extract_counts
from sCIN.assess import assess, assess_joint_from_separate_embs
from sklearn.model_selection import train_test_split
import anndata as ad
import pandas as pd
import numpy as np
import torch 
import random
import os
import argparse


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--mod1_file", type=str)
    parser.add_argument("--mod2_file", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--is_inv_metrics", action="store_true")
    parser.add_argument("--num_reps", type=int)  # max 10
    args = parser.parse_args()

    # Set seeds for replications
    seeds = [seed for seed in range(0, 100, 10)]
    if args.num_reps != 10:
        seeds = seeds[:args.num_reps]

    # Read data
    rna_adata = ad.read_h5ad(args.mod1_file)
    atac_adata = ad.read_h5ad(args.mod2_file)
    rna_counts, atac_counts = extract_counts(rna_adata, atac_adata)
    labels = rna_adata.obs["cell_type_encoded"].values

    res = []
    for i, seed in enumerate(seeds):

        # Control randomness
        torch.manual_seed(seeds)
        torch.random.manual_seed(seeds)
        torch.cuda.manual_seed_all(seeds)
        np.random.seed(seeds)
        random.seed(seeds)
        
        rep = i + 1

        # Train/test data split
        rna_train, rna_test, atac_train, atac_test, \
            labels_train, labels_test = train_test_split(rna_counts,
                                                         atac_counts,
                                                         labels,
                                                         test_size=0.3,
                                                         random_state=seed)
        
        
        train_dict = train_sCIN(rna_train, 
                                atac_train,
                                labels_train, 
                                settings=sCIN,
                                save_dir=args.save_dir,
                                rep=rep, 
                                is_pca=True)
        
        rna_embs, atac_embs = get_emb_sCIN(rna_test, 
                                           atac_test,
                                           train_dict=train_dict,
                                           rep=rep)

        # Save outputs
        embs_save_dir = os.path.join(args.save_dir, "embs")
        os.makedirs(embs_save_dir, exist_ok=True)
        rna_embs_df = pd.DataFrame(rna_embs)
        atac_embs_df = pd.DataFrame(atac_embs)
        labels_df = pd.DataFrame(labels_test)
        rna_embs_df.to_csv(os.path.join(embs_save_dir, f"rna_embs_rep{rep}.csv"), index=False)
        atac_embs_df.to_csv(os.path.join(embs_save_dir, f"atac_embs_rep{rep}.csv"), index=False)
        labels_df.to_csv(os.path.join(embs_save_dir, f"labels_test_rep{rep}.csv"), index=False)

        # Compute metrics
        recall_at_k_a2r, num_pairs_a2r, cell_type_acc_a2r, asw, medr_a2r = assess(atac_embs, 
                                                                                  rna_embs, 
                                                                                  labels_test, 
                                                                                  seed=seed)
        
        cell_type_acc_joint = assess_joint_from_separate_embs(mod1_embs=rna_embs,
                                                              mod2_embs=atac_embs,
                                                              labels=labels_test,
                                                              seed=seed)
        if args.is_inv_metrics:
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
    results.to_csv(os.path.join(args.save_dir, "outs", f"metrics_sCIN_{args.num_reps}reps.csv"), index=False)


if __name__ == "__main__":
    main()