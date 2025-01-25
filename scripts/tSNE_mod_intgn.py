import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import colorcet as cc
from sCIN.utils.utils import map_cell_types
import argparse
import os


def plot_tsne(data, labels, unique_colors, title, legend_title, save_path, legend_ncol,
              bbox_to_anchor, loc):

    plt.figure(figsize=(13, 5.5))
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(data[mask, 0], data[mask, 1], s=15, label=label, color=unique_colors[i])
    
    plt.tick_params(axis='both', which='major', labelsize=10)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(
        title=legend_title,
        bbox_to_anchor=bbox_to_anchor,  # bottom: (0.5, -0.1)  next left: (1.05, 0.5)
        loc=loc,
        fontsize=14,
        ncol=legend_ncol,
        frameon=False,
        markerscale=8,
        columnspacing=0.5,
        labelspacing=0.5 ,
        handletextpad=0.5
    )
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.clf()
  


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--unpaired", action="store_true")
    args = parser.parse_args()
    
    mapping_f_path = {
         "PBMC":"data/10x/mapped_cell_types.txt",
         "CITE":"data/cite/mapped_cell_types.txt",
         "SHARE":"data/share/mapped_cell_types.txt"}
    
    mapping = map_cell_types(path=mapping_f_path[args.data])
    map_func = np.vectorize(lambda x: mapping[x])
    
    if args.paired:

        BASE_DIR = f"results/{args.data}/{args.model}/V1/rep1/embs"
        SAVE_DIR = f"results/{args.data}/plots"
        if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR, exist_ok=True)
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
        mod_colors = {"Modality 1": "#2c7fb8", "Modality 2": "#d95f0e"}
        mod_unique_colors = [mod_colors[mod] for mod in np.unique(mod_lbls)]
        plot_tsne(joint_tsne, mod_lbls, mod_unique_colors,
                "t-SNE Colored by Modality", "Modalities",
                os.path.join(SAVE_DIR, "paired_embs_tsne_coloredby_mods.jpg"),
                legend_ncol=2)

        # t-SNE plot colored by cell types
        lbls = map_func(ct_lbls)
        lbls = np.hstack((lbls, lbls))
        print(lbls.shape)
        cell_types = np.unique(lbls)
        ct_colors = cc.glasbey[:len(cell_types)]
        plot_tsne(joint_tsne, lbls, ct_colors,
                "", "Cell Types", 
                os.path.join(SAVE_DIR, "paired_embs_tsne_coloredby_celltypes.jpg"), 
                legend_ncol=5)


    if args.unpaired:
        
        BASE_DIR = f"results/{args.data}/main_unpaired/rep1/p50/embs"
        SAVE_DIR = f"results/{args.data}/main_unpaired/plots"
        mod1_emb_path = os.path.join(BASE_DIR, "rna_emb0.npy")
        mod2_emb_path = os.path.join(BASE_DIR, "atac_emb0.npy")
        ct_lbls_path = os.path.join(BASE_DIR, "labels_test_0.npy")

        embs1 = np.load(mod1_emb_path) 
        embs2 = np.load(mod2_emb_path)
        ct_lbls = np.load(ct_lbls_path)
        m1 = np.repeat("RNA", len(embs1))
        m2 = np.repeat("ATAC", len(embs2))
        mod_lbls = np.hstack((m1, m2))

        print("First plotting starts ...")
        tsne = TSNE(n_components=2)
        joint = np.row_stack((embs1, embs2))
        joint_tsne = tsne.fit_transform(joint)

        # t-SNE plot colored by modalities
        mod_colors = {"RNA": "#7fc97f", "ATAC": "#beaed4"}
        mod_unique_colors = [mod_colors[mod] for mod in np.unique(mod_lbls)]
        plot_tsne(joint_tsne, mod_lbls, mod_unique_colors,
                "", "",
                os.path.join(SAVE_DIR, "upaired_embs_tsne_coloredby_mods_V2.jpg"),
                legend_ncol=1, bbox_to_anchor=(0.3, 0.9), loc="center")

        # t-SNE plot colored by cell types
        print("Second plotting starts ...")
        lbls = map_func(ct_lbls)
        lbls = np.hstack((lbls, lbls))
        print(lbls.shape)
        cell_types = np.unique(lbls)
        ct_colors = cc.glasbey[:len(cell_types)]
        plot_tsne(joint_tsne, lbls, ct_colors,
                "", "", 
                os.path.join(SAVE_DIR, "unpaired_embs_tsne_coloredby_celltypes_V2.jpg"), 
                legend_ncol=10, bbox_to_anchor=(12, 7.5), loc="center")
    
    print("Finished.")


if __name__ == "__main__":
    main()