"""
This is a script for plotting t-SNE embeddings of the original dataset and 
model's embeddings. (Figures 2-4e, f)

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
from sklearn.model_selection import train_test_split
import colorcet as cc
from sCIN.utils.utils import (plot_tsne_original, plot_tsne_embs, 
                                 extract_counts, map_cell_types,
                                 list_embs)
import argparse
import os


def plot_tsne_original(data:np.array, labels:np.array, save_dir:str,
                       file_type:str="png") -> None:
    
    tsne_ori = TSNE(n_components=2, learning_rate='auto',
                  init='random').fit_transform(data)
    
    cell_types = np.unique(labels)
    colors = cc.glasbey[:len(cell_types)]

    plt.figure(figsize=(12, 10))
    for i, cell_type in enumerate(cell_types):
          mask = labels == cell_type
          plt.scatter(tsne_ori[mask, 0], tsne_ori[mask, 1], 
                         s=10, label=f" {cell_type}", color=colors[i])
          
    plt.tick_params(axis='both', which='major', labelsize=8)
    ax = plt.gca()  
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(
    title="Cell Types", 
    bbox_to_anchor=(0.5, -0.2),  # 0.5, -0.2 for PBMC
    loc='center',          
    fontsize=10, 
    ncol=10,                     
    frameon=False,
    handleheight=1,           
    markerscale=4               
    )
    out = os.path.join(save_dir, f"tsne_original.{file_type}")
    plt.savefig(out, bbox_inches='tight')

    
def plot_tsne_embs(data:np.array, labels:np.array, save_dir:str, 
                   file_type:str="png") -> None:
    
    tsne_embs = TSNE(n_components=2, learning_rate='auto',
                  init='random').fit_transform(data)
    
    cell_types = np.unique(labels)
    colors = cc.glasbey[:len(cell_types)]

    plt.figure(figsize=(12, 10))
    for i, cell_type in enumerate(cell_types):
          mask = labels == cell_type
          plt.scatter(tsne_embs[mask, 0], tsne_embs[mask, 1], 
                         s=10, label=f" {cell_type}", color=colors[i])
          
    plt.tick_params(axis='both', which='major', labelsize=8)
    ax = plt.gca()  
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # plt.legend(
    # title="Cell Types", 
    # bbox_to_anchor=(0.5, -0.15),  
    # loc='center left',          
    # fontsize=8, 
    # ncol=6,                     
    # frameon=False,
    # handleheight=1,           
    # markerscale=6               
    # )
    out = os.path.join(save_dir, f"tsne_embs.{file_type}")
    plt.savefig(out)


def main() -> None:
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Name of the original data")
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    DATA = {
        "SHARE":("ata/share/Ma-2020-RNA.h5ad", 
                 "data/share/Ma-2020-ATAC.h5ad",
                 "data/share/mapped_cell_types.txt"),
        "PBMC":("data/10x/10x-Multiome-Pbmc10k-RNA.h5ad", 
                "data/10x/10x-Multiome-Pbmc10k-ATAC.h5ad",
                "data/10x/mapped_cell_types.txt"),
        "CITE":("data/cite/rna.h5ad", "data/cite/adt.h5ad",
                "data/cite/mapped_cell_types.txt")
    }
    EMBS_DIR = f"results/{args.data}/{args.model}/V1/rep1/embs"
    SAVE_DIR = f"results/{args.data}/plots"
    FILE_TYPE = "pdf"
    NUM_PC = 2000


    mod1_path, mod2_path, mapped_ct_path = DATA[args.data]

    mod1 = ad.read_h5ad(mod1_path)
    mod2 = ad.read_h5ad(mod2_path)
    mod1_counts, mod2_counts = extract_counts(mod1, mod2)
    lbls = mod1.obs["cell_type_encoded"]

    embs1 = np.load(os.path.join(EMBS_DIR, "rna_emb0.npy"))
    embs2 = np.load(os.path.join(EMBS_DIR, "atac_emb0.npy"))
    embs_lbls = np.load(os.path.join(EMBS_DIR, "labels_test_0.npy"))

    ct_mapping = map_cell_types(mapped_ct_path)
    map_func = np.vectorize(lambda x: ct_mapping[x])
    cell_types = map_func(lbls)

    joint_embs = np.column_stack((embs1, embs2))
    print("Plot for embeddings ...")
    plot_tsne_embs(joint_embs, embs_lbls, SAVE_DIR, FILE_TYPE)

    _, test1, _, test2, _, lbls_test = train_test_split(
        mod1_counts, mod2_counts, lbls, test_size=0.3,
        random_state=0
    )
    cell_types = map_func(lbls_test)
    joint_ori = np.column_stack((test1, test2))
    pca = PCA(n_components=NUM_PC)
    joint_ori_pca = pca.fit_transform(joint_ori)
    print("Plot for original data ...")
    plot_tsne_original(joint_ori_pca, cell_types, SAVE_DIR, FILE_TYPE)

    print("Finished.")


if __name__ == "__main__":
    main()