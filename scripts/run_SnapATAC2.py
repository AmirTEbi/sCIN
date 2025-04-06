import snapatac2 as snap
import numpy as np
import pandas as pd
import anndata as ad
from sCIN.assess import assess_joint
from ..sCIN import assess_joint
import argparse
import os


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_file")
    parser.add_argument("--atac_file")
    parser.add_argument("--save_dir")
    parser.add_argument("--num_reps", type=int, choices=range(1, 11), help="Number of replication(1-10)")
    args = parser.parse_args()

    rna_file = ad.read_h5ad(args.rna_file)
    atac_file = ad.read_h5ad(args.atac_file)
    labels = rna_file.obs["cell_type_encoded"].values

    assert (rna_file.obs_names == atac_file.obs_names).all()
    seeds = [seed for seed in range(0, 100, 10)]
    if args.num_reps != 10:
        seeds = seeds[:args.num_reps]
    res = []
    embs_save_dir = os.path.join(args.save_dir, "embs")
    os.makedirs(embs_save_dir, exist_ok=True)

    for i, seed in enumerate(seeds):
        rep = i + 1
        joint_embs = snap.tl.multi_spectral([rna_file, atac_file], features=None)[1]
        df = pd.DataFrame(joint_embs)
        df.to_csv(os.path.join(embs_save_dir, f"joint_embs_{rep}.csv"), 
                  index=False, header=False)
        
        cell_type_acc, asw = assess_joint(joint_embs, labels=labels,
                                          seed=seed, k=1)
        res.append({
            "Model":"SnapATAC2",
            "Replicates":rep,
            "cell_type_acc_joint":cell_type_acc,
            "cell_type_ASW":asw
        })
    
    results = pd.DataFrame(res)
    out_save_dir = os.path.join(args.save_dir, "outs")
    os.mkdir(out_save_dir, exist_ok=True)
    results.to_csv(os.path.join(out_save_dir, f"SnapATAC2_joint_embs_{args.num_reps}_reps.csv"), index=False)


if __name__ == "__main__":
    main()