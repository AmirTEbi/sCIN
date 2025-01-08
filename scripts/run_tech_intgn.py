import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import colorcet as cc
from sc_cool.utils.utils import map_cell_types
import argparse
import os


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--unpaired", action="store_true")
    args = parser.parse_args()

    SAVE_DIR = f"results/{args.model}/plots"
    if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR, exist_ok=True)
    
    mapping_f_path = {
         "PBMC":"data/10x/mapped_cell_types.txt",
         "CITE":"data/share/mapped_cell_types.txt",
         "SHARE":"data/cite/mapped_cell_types.txt"}
    mapping = map_cell_types(path=mapping_f_path[args.data])
    map_func = np.vectorize(lambda x: mapping[x])
    
    if args.paired:

        BASE_DIR = f"results/{args.data}/{args.model}/embs"
        mod1_emb_path = os.path.join(BASE_DIR, "rna_emb0.npy")
        mod2_emb_path = os.path.join(BASE_DIR, "atac_emb0.npy")
        ct_lbls_path = os.path.join(BASE_DIR, "labels_test_0.npy")

        embs1 = np.load(mod1_emb_path) 
        embs2 = np.load(mod2_emb_path)
        ct_lbls = np.load(ct_lbls_path)
        m1 = np.repeat("Modality 1", len(embs1))
        m2 = np.repeat("Modality 2", len(embs2))
        mod_lbls = np.hstack((m1, m2))
        
        tsne = TSNE(n_components=2)
        joint = np.vstack((embs1, embs2))
        joint_tsne = tsne.fit_transform(joint)

        # t-SNE plot colored by modalities
        mod_colors = {"Modality 1":"#2c7fb8",
                       "Modality 2":"#d95f0e"}
        for mod in mod_lbls:
             mask = mod_lbls == mod
             plt.scatter(joint_tsne[mask, 0], joint_tsne[mask, 1], 
                         s=0.5, label=f" {mod}", color=mod_colors[mod])

        plt.tick_params(axis='both', which='major', labelsize=8)
        ax = plt.gca()  
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.legend(
        title="", 
        bbox_to_anchor=(1.05, 1), 
        loc='upper left',
        fontsize=8, 
        ncol=2, 
        frameon=False,
        handleheight=1, 
        markerscale=6)
        plt.tight_layout()

        plt.savefig(os.path.join(SAVE_DIR, "paired_embs_tsne_coloredby_mods.eps"))

        # t-SNE plot colored by cell types
        lbls = map_func(ct_lbls)
        cell_types = np.unique(lbls)
        ct_colors = cc.glasbey[:len(cell_types)]

        for i, cell_type in enumerate(cell_types):
            mask = lbls == cell_type
            plt.scatter(joint_tsne[mask, 0], joint_tsne[mask, 1], 
                        s=0.5, label=f" {cell_type}", color=ct_colors[i])
            
        plt.tick_params(axis='both', which='major', labelsize=8)
        ax = plt.gca()  
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.legend(
        title="Cell Types", 
        bbox_to_anchor=(1.05, 1), 
        loc='upper left',
        fontsize=8, 
        ncol=2, 
        frameon=False,
        handleheight=1,  # Adjust spacing between markers
        markerscale=6)   # Adjust size of legend markers

        plt.savefig(os.path.join(SAVE_DIR, "paired_embs_tsne_coloredby_celltypes.eps"))


    if args.unpaired:
        
        BASE_DIR = f"results/{args.data}/main_unpaired/rep0/p50"
        mod1_emb_path = os.path.join(BASE_DIR, "rna_emb0.npy")
        mod2_emb_path = os.path.join(BASE_DIR, "atac_emb0.npy")
        ct_lbls_path = os.path.join(BASE_DIR, "labels_test_0.npy")

        embs1 = np.load(mod1_emb_path) 
        embs2 = np.load(mod2_emb_path)
        ct_lbls = np.load(ct_lbls_path)
        m1 = np.repeat("Modality 1", len(embs1))
        m2 = np.repeat("Modality 2", len(embs2))
        mod_lbls = np.hstack((m1, m2))

        tsne = TSNE(n_components=2)
        joint = np.vstack((embs1, embs2))
        joint_tsne = tsne.fit_transform(joint)

        # t-SNE plot colored by modalities
        mod_colors = {"Modality 1":"#2c7fb8", 
                      "Modality 2":"#d95f0e"}
        
        for mod in mod_lbls:
             mask = mod_lbls == mod
             plt.scatter(joint_tsne[mask, 0], joint_tsne[mask, 1], 
                         s=0.5, label=f" {mod}", color=mod_colors[mod])

        plt.tick_params(axis='both', which='major', labelsize=8)
        ax = plt.gca()  
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.legend(
        title="", 
        bbox_to_anchor=(1.05, 1), 
        loc='upper left',
        fontsize=8, 
        ncol=2, 
        frameon=False,
        handleheight=1, 
        markerscale=6)
        plt.tight_layout()

        plt.savefig(os.path.join(SAVE_DIR, "unpaired_embs_tsne_coloredby_mods.eps"))

        # t-SNE plot colored by cell types
        lbls = map_func(ct_lbls)
        cell_types = np.unique(lbls)
        ct_colors = cc.glasbey[:len(cell_types)]

        for i, cell_type in enumerate(cell_types):
            mask = lbls == cell_type
            plt.scatter(joint_tsne[mask, 0], joint_tsne[mask, 1], 
                        s=0.5, label=f" {cell_type}", color=ct_colors[i])
            
        plt.tick_params(axis='both', which='major', labelsize=8)
        ax = plt.gca()  
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.legend(
        title="Cell Types", 
        bbox_to_anchor=(1.05, 1), 
        loc='upper left',
        fontsize=8, 
        ncol=2, 
        frameon=False,
        handleheight=1,  # Adjust spacing between markers
        markerscale=6)   # Adjust size of legend markers

        plt.savefig(os.path.join(SAVE_DIR, "unpaired_embs_tsne_coloredby_celltypes.eps"))


if __name__ == "__main__":
    main()