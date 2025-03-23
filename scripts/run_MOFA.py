from ..models.MOFA import get_mofa_emb
from ..configs import MOFA
from ..sCIN.utils import extract_counts
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

        mod1_train, mod1_test, mod2_train, mod2_test, \
            _, labels_test = train_test_split(mod1_counts,
                                                         mod2_counts,
                                                         labels,
                                                         test_size=0.3,
                                                         random_state=seed)
        
        mod1_embs, mod2_embs = get_mofa_emb(
            mod1_train, mod1_test, mod2_train, mod2_test,
            settings=MOFA, seed=seed
        )

        # Save outputs
        mod1_embs_df = pd.DataFrame(mod1_embs)
        mod2_embs_df = pd.DataFrame(mod2_embs)
        labels_df = pd.DataFrame(labels_test)
        mod1_embs_df.to_csv(os.path.join(args.save_dir, "embs", f"mod1_embs_{i+1}.csv"), index=False)
        mod2_embs_df.to_csv(os.path.join(args.save_dir, "embs", f"mod2_embs_{i+1}.csv"), index=False)
        labels_df.to_csv(os.path.join(args.save_dir, "embs", f"labels_test_{i+1}.csv"), index=False)

        # Compute metrics
        recall_at_k_12, num_pairs_12, cell_type_acc_12, asw = assess(
            mod1_embs, mod2_embs, labels_test, seed=seed
        )

        recall_at_k_21, num_pairs_21, cell_type_acc_21, asw = assess(
            mod2_embs, mod1_embs, labels_test, seed=seed
        )

        for k, r_at_k12 in recall_at_k_12.items():
            r_at_k21 = recall_at_k_21.get(k, 0)
            res.append({
                "Models":"MOFA",
                "Replicates":i+1,
                "k":k,
                "Recall_at_k":r_at_k12,
                "Recall_at_k_inv":r_at_k21,
                "num_pairs":num_pairs_12,
                "num_pairs_inv":num_pairs_21,
                "cell_type_acc":cell_type_acc_12,
                "cell_type_acc_inv":cell_type_acc_21,
                "cell_type_ASW":asw
            })
    
    results = pd.DataFrame(res)
    results.to_csv(os.path.join(args.save_dir, "outs", f"metrics_MOFA_{args.num_reps}reps.csv"))


if __name__ == "__main__":
    main()