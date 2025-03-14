import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import silhouette_score
from sCIN.benchmarks.assess import compute_metrics
from sCIN.benchmarks.MedR import compute_MedR
import argparse
import re
import os


def extrac_seed(file):

    match = re.search(r'_(\d+)\.h5ad$', file)

    return int(match.group(1)) if match else None

def extract_files(dir, extract_func=extrac_seed) -> tuple[list, list]:

    rna_files = [f for f in os.listdir(dir) if "rna_bridge_integrated" in f]
    atac_files = [f for f in os.listdir(dir) if "atac_bridge_integrated" in f]
    rna_files_sorted = sorted(rna_files, key=extract_func)
    atac_files_sorted = sorted(atac_files, key=extract_func)

    return rna_files_sorted, atac_files_sorted


def compute_asw(data:np.array, labels:np.array) -> float:

    return (silhouette_score(data, labels) + 1) / 2


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--embs_dir", type=str, help="Directory of the embeddings.")
    parser.add_argument("--save_dir", type=str, help="Directory to save the results.")
    parser.add_argument("--atac_to_rna", action="store_true", help="Directory to save the results.")
    parser.add_argument("--rna_to_atac", action="store_true", help="Directory to save the results.")
    parser.add_argument("--two_sided", action="store_true", help="Directory to save the results.")
    args = parser.parse_args()

    res = []

    rna_files, atac_files = extract_files(args.embs_dir)
    print(len(rna_files))
    print(len(atac_files))

    for i in range(len(rna_files)):
        rna_path = rna_files[i]
        atac_path = atac_files[i]
        rna = ad.read_h5ad(os.path.join(args.embs_dir, rna_path))
        atac = ad.read_h5ad(os.path.join(args.embs_dir, atac_path))

        rna_embs = rna.obsm["Embedding"]
        atac_embs = atac.obsm["Embedding"]
        lbls = rna.obs["cell_type_encoded"].values

        joint = np.column_stack((rna_embs, atac_embs))
        asw = compute_asw(joint, lbls)

        if args.atac_to_rna:
            ratk, num_pairs, ct_acc = compute_metrics(atac_embs, rna_embs, lbls)
            medr = compute_MedR(atac_embs, rna_embs)
            for k, v in ratk.items():
                res.append({
                    "Models":["scBridge"],
                    "Replicates":[i+1],
                    "k":[k],
                    "Recall_at_k":[v],
                    "num_pairs":[num_pairs],
                    "cell_type_acc":[ct_acc],
                    "cell_type_ASW":[asw],
                    "MedR":[medr]
                })

        elif args.rna_to_atac:
            ratk, num_pairs, ct_acc = compute_metrics(rna_embs, atac_embs, lbls)
            medr = compute_MedR(rna_embs, atac_embs)
            for k, v in ratk.items():
                res.append({
                    "Models":["scBridge"],
                    "Replicates":[i+1],
                    "k":[k],
                    "Recall_at_k":[v],
                    "num_pairs":[num_pairs],
                    "cell_type_acc":[ct_acc],
                    "cell_type_ASW":[asw],
                    "MedR":[medr]
                })

        elif args.two_sided:
            ratk_a2r, num_pairs_a2r, ct_acc_a2r = compute_metrics(atac_embs, rna_embs, lbls)
            medr_a2r = compute_MedR(atac_embs, rna_embs)
            ratk_r2a, num_pairs_r2a, ct_acc_r2a = compute_metrics(rna_embs, atac_embs, lbls)
            medr_r2a = compute_MedR(rna_embs, atac_embs)

            for v_a, k_a in ratk_a2r.items():
                v_r = ratk_r2a.get(k_a, 0)
                res.append({
                    "Models":["scBridge"],
                    "Replicates":[i+1],
                    "k":[k_a],
                    "Recall_at_k_r2a":[v_a],
                    "Recall_at_k_a2r":[v_r],
                    "num_pairs_a2r":[num_pairs_a2r],
                    "num_pairs_r2a":[num_pairs_r2a],
                    "cell_type_acc_r2a":[ct_acc_a2r],
                    "cell_type_acc_a2r":[ct_acc_r2a],
                    "cell_type_ASW":[asw],
                    "MedR_a2r":[medr_a2r],
                    "MedR_r2a":[medr_r2a]
                })

        result = pd.DataFrame(res)
    result.to_csv(os.path.join(args.save_dir, "scBridge_metrics"), index=False)

    print("Finished.")


if __name__ == "__main__":
    main()