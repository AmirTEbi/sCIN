from sCIN.assess import compute_graph_connectivity, cell_type_at_k_unpaired
import numpy as np 
import pandas as pd 
import argparse
import re
import os 

DIRS = {
    "SHARE":{
        "sCIN": "results/SHARE/V1-embs",
        "Con-AAE": "results/SHARE/Con-AAE/embs",
        "MOFA+": "results/SHARE/MOFA/embs",
        "Harmony": "results/SHARE/Harmony/embs",
        "AE": "results/SHARE/AE/embs", 
        "scGLUE": "results/SHARE/scGLUE/embs",
        "scBridge": "results/SHARE/scBridge",
        "sciCAN": "results/SHARE/sciCAN/embs"
    },
    "PBMC":{
        "sCIN": "results/PBMC/sCIN/V1-embs",
        "Con-AAE": "results/PBMC/Con-AAE/embs",
        "MOFA+": "results/PBMC/MOFA/embs",
        "Harmony": "results/PBMC/Harmony/embs",
        "AE": "results/PBMC/AE/embs", 
        "scGLUE": "results/PBMC/scGLUE/embs",
        "scBridge": "results/PBMC/scBridge/embs",
        "sciCAN": "results/PBMC/sciCAN/embs"
    },
    "CITE":{
        "sCIN": "results/CITE/sCIN/V1-embs",
        "Con-AAE": "results/CITE/Con-AAE/embs",
        "MOFA+": "results/CITE/MOFA/embs",
        "Harmony": "results/CITE/Harmony/embs",
        "AE": "results/CITE/AE/embs", 
        "scGLUE": "results/CITE/scGLUE/embs",
        "sciCAN": "results/CITE/sciCAN/embs"
    }
}


def setup_args():

    parser = argparse.ArgumentParser()

    return parser


def load_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.csv':
        return pd.read_csv(filepath)
    elif ext == '.npy':
        return np.load(filepath, allow_pickle=True)
    else:
        raise ValueError(f"Unsupported file extension '{ext}' for file {filepath}")


def main():
    parser = setup_args()
    args   = parser.parse_args()

    datasets = [args.dataset] if args.dataset else DIRS.keys()
    for data in datasets:
        all_results = []
        for model, embs_dir in DIRS[data].items():
            print(f"Processing dataset={data}, model={model}")

            try:
                files = os.listdir(embs_dir)
            except FileNotFoundError:
                print(f"Warning: directory '{embs_dir}' not found, skipping.")
                continue

            rna_files   = sorted(f for f in files if 'rna'   in f.lower())
            atac_files  = sorted(f for f in files if 'atac'  in f.lower())
            label_files = sorted(f for f in files if 'label' in f.lower())

            # Build maps by integer ID
            def build_map(file_list):
                mp = {}
                for fname in file_list:
                    m = re.search(r"(\d+)", fname)
                    if m:
                        mp[int(m.group())] = fname
                return mp

            atac_map  = build_map(atac_files)
            label_map = build_map(label_files)

            for file_idx, rna_file in enumerate(rna_files, start=1):
                m = re.search(r"(\d+)", rna_file)
                if not m:
                    continue
                id_ = int(m.group())

                atac_match  = atac_map .get(id_)
                label_match = label_map.get(id_)

                if not (atac_match and label_match):
                    missing = []
                    if not atac_match:  missing.append('ATAC')
                    if not label_match: missing.append('label')
                    print(f"\tWarning: Missing {', '.join(missing)} for RNA index {id_}")
                    continue

                # Full paths
                rna_path   = os.path.join(embs_dir, rna_file)
                atac_path  = os.path.join(embs_dir, atac_match)
                label_path = os.path.join(embs_dir, label_match)

                # Load
                rna_embs  = load_file(rna_path)
                atac_embs = load_file(atac_path)
                labels    = load_file(label_path)

                # Compute metrics
                joint_embs = np.concatenate([rna_embs, atac_embs], axis=1)
                gc = compute_graph_connectivity(joint_embs, labels)

                ct_at_k_a2r = cell_type_at_k_unpaired(atac_embs, rna_embs, labels, labels)
                ct_at_k_r2a = cell_type_at_k_unpaired(rna_embs, atac_embs, labels, labels)

                # Collect
                for k, v_a2r in ct_at_k_a2r.items():
                    v_r2a = ct_at_k_r2a.get(k, np.nan)
                    all_results.append({
                        "Model":model,
                        "Replicate":file_idx,
                        "k":k,
                        "cell_type_a2r":v_a2r,
                        "cell_type_r2a":v_r2a,
                        "GC_joint":gc
                    })

        # Save per‐dataset
        df = pd.DataFrame(all_results)
        outdir = os.path.join("sCIN/results", data, "CtatK-GC-Paired")
        os.makedirs(outdir, exist_ok=True)
        df.to_csv(os.path.join(outdir, f"cell-type-at-k-GC-Paired-{data}.csv"), index=False)
        print(f"Saved results for {data} → {outdir}")


if __name__ == "__main__":
    main()