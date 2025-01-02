"""
This is a script for plotting t-SNE embeddings of the original dataset and 
model's embeddings. (figures e and f)

To run:

> cd sc-cool
** make sure to activate the virtual env before:
    - source .venv/bin/activate (OS X and Linux)
    - source .venv/Scripts/activate.ps1 (Windows)

sc-cool > python scripts/original_embs_tsne_plt.py --data "..." --embs_path "..." --embs_dir "..." --save_dir "..."
"""

import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import colorcet as cc
from sc_cool.utils.utils import (plot_tsne_original, plot_tsne_embs, 
                                 extract_counts, map_cell_types,
                                 list_embs)
import argparse
import os


def main() -> None:
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Name of the original data")
    parser.add_argument("--embs_path", type=str, help="Path to the embeddings file")
    parser.add_argument("--embs_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()

    if args.data == "share":
        mod1 = ad.read_h5ad("data/share/Ma-2020-RNA.h5ad")
        mod2 = ad.read_h5ad("data/share/Ma-2020-ATAC.h5ad")
        mapped_ct_path = "data/share/mapped_cell_types.txt"
    
    elif args.data == "pbmc":
        mod1 = ad.read_h5ad("data/10x/10x-Multiome-Pbmc10k-RNA.h5ad")
        mod2 = ad.read_h5ad("data/10x/10x-Multiome-Pbmc10k-ATAC.h5ad")
        mapped_ct_path = "data/10x/mapped_cell_types.txt"

    elif args.data == "cite":
        mod1 = ad.read_h5ad("data/cite/rna.h5ad")
        mod2 = ad.read_h5ad("data/cite/adt.h5ad")
        mapped_ct_path = "data/cite/mapped_cell_types.txt"

    print("Plotting original data ...")
    mod1_counts, mod2_counts = extract_counts(mod1, mod2)
    print(f"Shape of the first modality: {mod1_counts.shape}")
    print(f"Shape of the second modality: {mod2_counts.shape}")
    lbls = mod1.obs["cell_type_encoded"]

    ct_mapping = map_cell_types(mapped_ct_path)
    map_func = np.vectorize(lambda x: ct_mapping[x])
    cell_types = map_func(lbls)

    joint = np.hstack((mod1_counts, mod2_counts))
    pca = PCA(n_components=2000)
    joint_pca = pca.fit_transform(joint)

    unique_ct = np.unique(cell_types)
    plot_tsne_original(data=joint_pca, labels=unique_ct,
                       is_show=False, save_dir=args.save_dir)
    
    print("Plotting embeddings ...")
    paths = list_embs(embs_dir=args.embs_dir, seeds=[0], is_ct=True)
    path1, path2, path_lbls = paths[0]
    embs1 = np.load(path1)
    embs2 = np.load(path2)
    embs_lbls = np.load(path_lbls)
    embs_cell_types = map_func(embs_lbls)
    joint_embs = np.hstack((embs1, embs2))
    print(f"Shape of the joint embs: {joint_embs.shape}")
    unique_embs_ct = np.unique(embs_cell_types)
    plot_tsne_embs(joint_embs=joint_embs, labels=unique_embs_ct,
                   is_show=False, save_dir=args.save_dir)
    
    print("Finished.")


if __name__ == "__main__":
    main()