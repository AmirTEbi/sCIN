import snapatac2 as snap
import numpy as np
import pandas as pd
import anndata as ad
from sCIN.assess import assess_joint
from sCIN.utils import setup_logging
from scipy.sparse import csr_matrix, issparse
import argparse
import logging
import os


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_file")
    parser.add_argument("--atac_file")
    parser.add_argument("--save_dir")
    parser.add_argument("--num_reps", type=int, choices=range(1, 11), help="Number of replication(1-10)")
    args = parser.parse_args()
    
    log_save_dir = os.path.join(args.save_dir, "logs")
    os.makedirs(log_save_dir, exist_ok=True)
    setup_logging(level="info", log_dir=log_save_dir, model_name="SnapATAC2")

    rna_file = ad.read_h5ad(args.rna_file)
    if np.any(~np.isfinite(rna_file.X.data)):
        raise ValueError("rna_file.X contains NaN or Inf values.")
    if not isinstance(rna_file.X, csr_matrix):
        print("Converting rna_file.X to csr_matrix")
        rna_file.X = csr_matrix(rna_file.X)
    
    atac_file = ad.read_h5ad(args.atac_file)
    if np.any(~np.isfinite(atac_file.X.data)):
        raise ValueError("atac_file.X contains NaN or Inf values.")
    if not isinstance(atac_file.X, csr_matrix):
        print("Converting atac_file.X to csr_matrix")
        atac_file.X = csr_matrix(atac_file.X)
    
    print("rna_file.X density:", rna_file.X.nnz / (rna_file.X.shape[0] * rna_file.X.shape[1]))
    print("atac_file.X density:", atac_file.X.nnz / (atac_file.X.shape[0] * atac_file.X.shape[1]))
    
    # print(type(rna_file.X))
    # print(type(atac_file.X))

    labels = rna_file.obs["cell_type_encoded"].values
    logging.info("Files loaded.")

    assert (rna_file.obs_names == atac_file.obs_names).all()
    seeds = [seed for seed in range(0, 100, 10)]
    if args.num_reps != 10:
        seeds = seeds[:args.num_reps]
    res = []
    embs_save_dir = os.path.join(args.save_dir, "embs")
    os.makedirs(embs_save_dir, exist_ok=True)

    for i, seed in enumerate(seeds):
        rep = i + 1
        logging.info(f"Replication {rep}")
        joint_embs = snap.tl.multi_spectral([rna_file, atac_file], features=None)[1]
        logging.info("Embeddings generated.")
        df = pd.DataFrame(joint_embs)
        df.to_csv(os.path.join(embs_save_dir, f"joint_embs_{rep}.csv"), 
                  index=False, header=False)
        logging.info(f"Embeddings save to {embs_save_dir}")
        
        cell_type_acc, asw = assess_joint(joint_embs, labels=labels,
                                          seed=seed, k=1)
        logging.info("Metrics has been computed.")
        res.append({
            "Model":"SnapATAC2",
            "Replicates":rep,
            "cell_type_acc_joint":cell_type_acc,
            "cell_type_ASW":asw
        })
    
    results = pd.DataFrame(res)
    out_save_dir = os.path.join(args.save_dir, "outs")
    os.makedirs(out_save_dir, exist_ok=True)
    results.to_csv(os.path.join(out_save_dir, f"SnapATAC2_joint_embs_{args.num_reps}_reps.csv"), index=False)
    logging.info(f"Results saved to {out_save_dir}")


if __name__ == "__main__":
    main()