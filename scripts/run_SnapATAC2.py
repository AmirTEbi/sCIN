import snapatac2 as snap
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import silhouette_score
from ..sCIN.assess import assess_joint
from sCIN.benchmarks.MedR import compute_MedR
from ..sCIN import assess_joint
import argparse
import re
import os


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--mod1_file")
    parser.add_argument("--mod2_file")
    parser.add_argument("--save_dir")
    args = parser.parse_args()

    mod1_file = ad.read_h5ad(args.mod1_file)
    mod2_file = ad.read_h5ad(args.mod2_file)
    labels = mod1_file.obs["cell_type_encoded"].values

    assert (mod1_file.obs_names == mod2_file.obs_names).all()
    seeds = [seed for seed in range(0, 100, 10)]
    res = []
    for i, seed in enumerate(seeds):
        joint_embs = snap.tl.multi_spectral([mod1_file, mod2_file], features=None)[1]
        df = pd.DataFrame(joint_embs)
        df.to_csv(os.path.join(args.save_dir, "embs", f"joint_embs_{i + 1}.csv"), 
                  index=False, header=False)
        
        cell_type_acc, asw = assess_joint(joint_embs, labels=labels,
                                          seed=seed, k=1)
        res.append({
            "Model":"SnapATAC2",
            "Replicates":i,
            "cell_type_acc":cell_type_acc,
            "cell_type_ASW":asw
        })
    
    results = pd.DataFrame(res)
    results.to_csv(os.path.join(args.save_dir, "outs", "SnapATAC2_joint_embs_10_reps.csv"), index=False)

if __name__ == "__main__":
    main()