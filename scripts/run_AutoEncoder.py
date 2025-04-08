from ..models.AutoEncoder import train_AutoEncoder, get_emb_AutoEncoder
from ..configs import AutoEncoder
from ..sCIN.utils import extract_counts, append_rows_has_inverse
from ..sCIN.assess import assess
from sklearn.model_selection import train_test_split
import anndata as ad
import pandas as pd
import os
import argparse


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--mod1_file")
    parser.add_argument("--mod2_file")
    parser.add_argument("--num_reps", type=int)  # max 10
    args = parser.parse_args()

    # Set seeds for replications
    seeds = [seed for seed in range(0, 100, 10)]
    if args.num_reps != 10:
        seeds = seeds[:args.num_reps]

    # Read data
    mod1_adata = ad.read_h5ad(args.mod1_file)
    mod2_adata = ad.read_h5ad(args.mod2_file)
    mod1_counts, mod2_counts = extract_counts(mod1_adata, mod2_adata)
    labels = mod1_adata.obs["cell_type_encoded"].values

    res = []
    for i, seed in enumerate(seeds):
        
        rep = i + 1
        mod1_train, mod1_test, mod2_train, mod2_test, \
            _, labels_test = train_test_split(mod1_counts,
                                              mod2_counts,
                                              labels,
                                              test_size=0.3,
                                              random_state=seed)
        
        # train
        train_dict = train_AutoEncoder(mod1_train, mod2_train,
                                       settings=AutoEncoder, is_pca=True,
                                       rep=rep)
        
        # Get embeddings
        mod1_embs, mod2_embs = get_emb_AutoEncoder(mod1_test, mod2_test,
                                                   labels_test=labels_test,
                                                   train_dict=train_dict)

        # Save outputs
        mod1_embs_df = pd.DataFrame(mod1_embs)
        mod2_embs_df = pd.DataFrame(mod2_embs)
        labels_df = pd.DataFrame(labels_test)
        mod1_embs_df.to_csv(os.path.join(args.save_dir, "embs", f"mod1_embs_{rep}.csv"), index=False)
        mod2_embs_df.to_csv(os.path.join(args.save_dir, "embs", f"mod2_embs_{rep}.csv"), index=False)
        labels_df.to_csv(os.path.join(args.save_dir, "embs", f"labels_test_{rep}.csv"), index=False)

        # Compute metrics
        recall_at_k_12, num_pairs_12, cell_type_acc_12, asw = assess(
            mod1_embs, mod2_embs, labels_test, seed=seed
        )

        recall_at_k_21, num_pairs_21, cell_type_acc_21, asw = assess(
            mod2_embs, mod1_embs, labels_test, seed=seed
        )

        res = append_rows_has_inverse(res, rep, recall_at_k_12, num_pairs_12,
                                      cell_type_acc_12, asw, recall_at_k_21,
                                      num_pairs_21, cell_type_acc_21)
    
    results = pd.DataFrame(res)
    results.to_csv(os.path.join(args.save_dir, "outs", f"metrics_AutoEncoder_{args.num_reps}reps.csv"))


if __name__ == "__main__":
    main()