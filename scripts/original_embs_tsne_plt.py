"""
This is a script for plotting t-SNE embeddings of the original dataset and 
model's embeddings. (Figures 2-4e, f)

To run:

> cd sc-cool
** make sure to activate the virtual env before:
    - source .venv/bin/activate (OS X and Linux)
    - source .venv/Scripts/activate.ps1 (Windows)

sc-cool > python scripts/original_embs_tsne_plt.py --data ... --model ...
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


def compute_tsne_ori(data:np.array, n_comps=2, lr="auto",
                     init="random") -> np.array:
     
     return TSNE(n_components=n_comps, learning_rate=lr,
                 init=init).fit_transform(data)

def compute_tsne_embs(joint_embs:np.array, n_comps=2, lr="auto",
                     init="random") -> np.array:
     
     return TSNE(n_components=n_comps, learning_rate=lr,
                 init=init).fit_transform(joint_embs)


def plot_tsne_original(data:np.array, labels:np.array, save_dir:str,
                       file_type:str="png") -> None:
    
    cell_types = np.unique(labels)
    colors = cc.glasbey[:len(cell_types)]

    plt.figure(figsize=(13, 4.25))
    for i, cell_type in enumerate(cell_types):
          mask = labels == cell_type
          plt.scatter(data[mask, 0], data[mask, 1], 
                         s=0.5, label=f" {cell_type}", color=colors[i])
          
    plt.tick_params(axis='both', which='major', labelsize=14)
    ax = plt.gca()  
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(top=1.25, bottom=0.5, hspace=0)
    plt.legend(
    title="",
    title_fontsize=16,
    bbox_to_anchor=(0.5, -0.2),  # 0.5, -0.2 for PBMC
    loc='center',          
    fontsize=14, 
    ncol=11,                     
    frameon=False,
    handleheight=1,           
    markerscale=8,
    columnspacing=0.5,
    labelspacing=0.5               
    )
    out = os.path.join(save_dir, f"tsne_original_V7.{file_type}")
    plt.savefig(out, bbox_inches='tight', pad_inches=0)

    
def plot_tsne_embs(data:np.array, labels:np.array, save_dir:str, 
                   file_type:str="png") -> None:
    
    cell_types = np.unique(labels)
    colors = cc.glasbey[:len(cell_types)]

    plt.figure(figsize=(13, 4.25))
    for i, cell_type in enumerate(cell_types):
          mask = labels == cell_type
          plt.scatter(data[mask, 0], data[mask, 1], 
                         s=0.5, label=f" {cell_type}", color=colors[i])
          
    plt.tick_params(axis='both', which='major', labelsize=14)
    ax = plt.gca()  
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(top=1.25, bottom=0.5, hspace=0)
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
    out = os.path.join(save_dir, f"tsne_embs_V7.{file_type}")
    plt.savefig(out, bbox_inches="tight", pad_inches=0)


def main() -> None:
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Name of the original data")
    parser.add_argument("--model", type=str)
    parser.add_argument("--tsne_save_dir", type=str)
    parser.add_argument("--tsne_plot_all", action="store_true")
    parser.add_argument("--tsne_plot_embs", action="store_true")
    parser.add_argument("--tsne_plot_ori", action="store_true")
    parser.add_argument("--plot_embs", action="store_true")
    parser.add_argument("--plot_ori", action="store_true")
    parser.add_argument("--plot_all", action="store_true")
    parser.add_argument("--tsne_embs", action="store_true")
    parser.add_argument("--tsne_ori", action="store_true")
    parser.add_argument("--tsne_all", action="store_true")
    args = parser.parse_args()

    DATA = {
        "SHARE":("data/share/Ma-2020-RNA.h5ad", 
                 "data/share/Ma-2020-ATAC.h5ad",
                 "data/share/mapped_cell_types.txt"),
        "PBMC":("data/10x/10x-Multiome-Pbmc10k-RNA.h5ad", 
                "data/10x/10x-Multiome-Pbmc10k-ATAC.h5ad",
                "data/10x/mapped_cell_types.txt"),
        "CITE":("data/cite/rna.h5ad", "data/cite/adt.h5ad",
                "data/cite/mapped_cell_types.txt")
    }
    EMBS_DIR = f"results/{args.data}/{args.model}/V1/rep1/embs"
    PLOT_SAVE_DIR = f"results/{args.data}/plots"
    if not os.path.exists(PLOT_SAVE_DIR):
        os.makedirs(PLOT_SAVE_DIR)
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
    if args.data == "SHARE":
        cell_types = np.where(
        cell_types == "ahigh CD34+ bulge",
        "a high CD34+ bulge",  
        np.where(
            cell_types == "alow CD34+ bulge",
            "a low CD34+ bulge",  
            cell_types))
    
    if args.tsne_plot_all:
        joint_embs = np.column_stack((embs1, embs2))
        tsne_embs = compute_tsne_embs(joint_embs)
        print("Plot for embeddings ...")
        plot_tsne_embs(tsne_embs, embs_lbls, PLOT_SAVE_DIR, FILE_TYPE)

        _, test1, _, test2, _, lbls_test = train_test_split(
            mod1_counts, mod2_counts, lbls, test_size=0.3,
            random_state=0
        )
        cell_types = map_func(lbls_test)
        if args.data == "SHARE":
            cell_types = np.where(
            cell_types == "ahigh CD34+ bulge",
            "a high CD34+ bulge",  
            np.where(
                cell_types == "alow CD34+ bulge",
                "a low CD34+ bulge",  
                cell_types))
        joint_ori = np.column_stack((test1, test2))
        pca = PCA(n_components=NUM_PC)
        joint_ori_pca = pca.fit_transform(joint_ori)
        tsne_ori = plot_tsne_original(joint_ori_pca)
        print("Plot for original data ...")
        plot_tsne_original(tsne_ori, cell_types, PLOT_SAVE_DIR, FILE_TYPE)

    elif args.tsne_plot_embs:
        joint_embs = np.column_stack((embs1, embs2))
        tsne_embs = compute_tsne_embs(joint_embs)
        plot_tsne_embs(tsne_embs, embs_lbls, PLOT_SAVE_DIR, FILE_TYPE)
         
    elif args.tsne_plot_ori:
        _, test1, _, test2, _, lbls_test = train_test_split(
            mod1_counts, mod2_counts, lbls, test_size=0.3,
            random_state=0
        )
        cell_types = map_func(lbls_test)
        if args.data == "SHARE":
            cell_types = np.where(
            cell_types == "ahigh CD34+ bulge",
            "a high CD34+ bulge",  
            np.where(
                cell_types == "alow CD34+ bulge",
                "a low CD34+ bulge",  
                cell_types))
        joint_ori = np.column_stack((test1, test2))
        pca = PCA(n_components=NUM_PC)
        joint_ori_pca = pca.fit_transform(joint_ori)
        tsne_ori = compute_tsne_ori(joint_ori_pca)
        plot_tsne_original(tsne_ori, cell_types, PLOT_SAVE_DIR, FILE_TYPE)
         
    elif args.plot_embs:
         tsne_embs = np.load(os.path.join(args.tsne_save_dir, "tsne_embs.npy"))
         plot_tsne_embs(tsne_embs, embs_lbls, PLOT_SAVE_DIR, FILE_TYPE)
      
    elif args.plot_ori:
         tsne_ori = np.load(os.path.join(args.tsne_save_dir, "tsne_ori.npy"))
         cell_types = map_func(lbls_test)
         if args.data == "SHARE":
            cell_types = np.where(
            cell_types == "ahigh CD34+ bulge",
            "a high CD34+ bulge",  
            np.where(
                cell_types == "alow CD34+ bulge",
                "a low CD34+ bulge",  
                cell_types))
         plot_tsne_original(tsne_ori, cell_types, PLOT_SAVE_DIR, FILE_TYPE)
         
    elif args.plot_all:
        tsne_embs = np.load(os.path.join(args.tsne_save_dir, "tsne_embs.npy"))
        plot_tsne_embs(tsne_embs, embs_lbls, PLOT_SAVE_DIR, FILE_TYPE)

        _, test1, _, test2, _, lbls_test = train_test_split(
            mod1_counts, mod2_counts, lbls, test_size=0.3,
            random_state=0
        )
        tsne_ori = np.load(os.path.join(args.tsne_save_dir, "tsne_ori.npy"))
        cell_types = map_func(lbls_test)
        if args.data == "SHARE":
            cell_types = np.where(cell_types == "ahigh CD34+ bulge", 
                                  "a high CD34+ bulge",  
                                  np.where(cell_types == "alow CD34+ bulge", 
                                           "a low CD34+ bulge", cell_types))
        plot_tsne_original(tsne_ori, cell_types, PLOT_SAVE_DIR, FILE_TYPE)
   
    elif args.tsne_embs:
        joint_embs = np.column_stack((embs1, embs2))
        tsne_embs = compute_tsne_embs(joint_embs)
        np.save(os.path.join(args.tsne_save_dir, "tsne_embs.npy"), tsne_embs)
    
    elif args.tsne_ori:
        _, test1, _, test2, _, lbls_test = train_test_split(
            mod1_counts, mod2_counts, lbls, test_size=0.3,
            random_state=0
        )
        joint_ori = np.column_stack((test1, test2))
        pca = PCA(n_components=NUM_PC)
        joint_ori_pca = pca.fit_transform(joint_ori)
        tsne_ori = compute_tsne_ori(joint_ori_pca)
        np.save(os.path.join(args.tsne_save_dir, "tsne_ori.npy"), tsne_ori)
         
    elif args.tsne_all:
        joint_embs = np.column_stack((embs1, embs2))
        tsne_embs = compute_tsne_embs(joint_embs)
        np.save(os.path.join(args.tsne_save_dir, "tsne_embs.npy"), tsne_embs)

        _, test1, _, test2, _, lbls_test = train_test_split(
            mod1_counts, mod2_counts, lbls, test_size=0.3,
            random_state=0
        )
        joint_ori = np.column_stack((test1, test2))
        pca = PCA(n_components=NUM_PC)
        joint_ori_pca = pca.fit_transform(joint_ori)
        tsne_ori = compute_tsne_ori(joint_ori_pca)
        np.save(os.path.join(args.tsne_save_dir, "tsne_ori.npy"), tsne_ori)
         
    print("Finished.")


if __name__ == "__main__":
    main()