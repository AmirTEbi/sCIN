"""
This is a script to compare ASW metric between cool integrated embeddings, PCA integrated embeddings, and original datasets.
"""

import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import umap
import colorcet as cc
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sc_cool.utils.utils import load_data
from sklearn.metrics import silhouette_score
import argparse


def main():

    parser = argparse.ArgumentParser(description='Get Options')
    parser.add_argument("--data", type=str, default="SHARE,PBMC,CITE", help="Names of the dataset")
    parser.add_argument("--our_asw", type=str, help="Mean ASWs of our embeddings across 10 replications for each dataset ")
    #parser.add_argument("--mod_labels", type=str, help="Labels for data modalities")
    parser.add_argument("--save_dir", type=str, help="Save directory")
    args = parser.parse_args()

    args.data = args.data.split(",")
    args.our_asw = list(map(float, args.our_asw.split(",")))
    #args.mod_labels = args.mod_labels.split(",")


    res_df = pd.DataFrame(columns=["Data", "PCA_ASW", "Mod1_ASW", "Mod2_ASW", "Our_ASW"])

    datasets = {
            "SHARE": {
                "mod1_path": "data/share/Ma-2020-RNA.h5ad",
                "mod2_path": "data/share/Ma-2020-ATAC.h5ad",
                "mapping_file": "data/share/mapped_cell_types.txt"
            },
            "PBMC": {
                "mod1_path": "data/10x/10x-Multiome-Pbmc10k-RNA.h5ad",
                "mod2_path": "data/10x/10x-Multiome-Pbmc10k-ATAC.h5ad",
                "mapping_file": "data/10x/mapped_cell_types.txt"
            },
            "CITE": {
                "mod1_path": "data/cite/rna.h5ad",
                "mod2_path": "data/cite/adt.h5ad",
                "mapping_file": "data/cite/mapped_cell_types.txt"
            }
        }
    
    pca_comps = {
        "SHARE":256,
        "PBMC": 256,
        "CITE": 32
    }
    
    for data, asw in zip(args.data, args.our_asw):

        if data in datasets:
            mod1 = ad.read_h5ad(datasets[data]["mod1_path"])
            mod2 = ad.read_h5ad(datasets[data]["mod2_path"])
            with open(datasets[data]["mapping_file"], "r") as f:
                ct_mapping = {int(k): v.strip() for k, v in (line.split(":", 1) for line in f)}
        else:
            raise ValueError(f"Dataset {data} is not recognized.")

        lbls = mod1.obs["cell_type_encoded"].values

        print(f"Shape of the first modality is: {mod1.shape}")
        print(f"Shape of the second modality is: {mod2.shape}")

        # Map encoded labels to the original labels
        map_func = np.vectorize(lambda x: ct_mapping[x])
        cell_types = map_func(lbls)
        assert cell_types.shape[0] == mod1.shape[0], "Mismatch between label shape and data shape."

        # Compute PCA joint embeddings
        mod1_counts = mod1.layers["norm_raw_counts"]
        mod2_counts = mod2.layers["norm_raw_counts"]
        joint = np.hstack((mod1_counts, mod2_counts))

        pca_joint = PCA(n_components=pca_comps[data])
        joint_pca = pca_joint.fit_transform(joint)
        print(f"Shape of the joint PCA embeddings is: {joint_pca.shape}")

        # Mod1 and Mod2-only PCA embeddings
        pca_mod1 = PCA(n_components=pca_comps[data])
        mod1_pca = pca_mod1.fit_transform(mod1_counts)
        print(f"Shape of the Mod1 PCA embeddings is: {mod1_pca.shape}")

        pca_mod2 = PCA(n_components=pca_comps[data])
        mod_2_pca = pca_mod2.fit_transform(mod2_counts)
        print(f"Shape of the Mod2 PCA embeddings is: {mod_2_pca.shape}")

        # Compute normalized ASW to be in [0,1]
        asw_pca_intg = (silhouette_score(joint_pca, lbls) + 1) / 2
        asw_mod1 = (silhouette_score(mod1_pca, lbls) + 1) / 2
        asw_mod2 = (silhouette_score(mod_2_pca, lbls) + 1) / 2
        print(f"ASW value for PCA integration: {asw_pca_intg}")
        print(f"ASW value for Mod1 {asw_mod1}")
        print(f"ASW value for Mod2 {asw_mod2}")

        new_row = pd.DataFrame({
            "Data":data,
            "PCA_ASW":asw_pca_intg,
            "Mod1_ASW":asw_mod1,
            "Mod2_ASW":asw_mod2,
            "Our_ASW":asw
        })

        res_df = pd.concat([res_df, new_row], ignore_index=True)

    os.makedirs(args.save_dir, exist_ok=True)
    output_file = os.path.join(args.save_dir, "Ours_PCA_Mod1_Mod2_ASW.csv")
    res_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")       

    # Plot
    data_lbls = tuple(args.data)
    mod_lbls = {
        "Modality 1": tuple(res_df["Mod1_ASW"].values),
        "Modality 2": tuple(res_df["Mod2_ASW"].values),
        "PCA integration": tuple(res_df["PCA_ASW"].values),
        "Our integration": tuple(res_df["Our_ASW"].values)
    }

    x = np.arange(len(data_lbls))
    width = 0.25
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    colors = sns.color_palette("colorblind", len(mod_lbls))
    for (attribute, measurement), color in zip(mod_lbls.items(), colors):
         offset = width * multiplier
         rects = ax.bar(x + offset, measurement, width, label=attribute, color=color)
         ax.bar_label(rects, padding=3)
         multiplier += 1

    max_asw = max(max(res_df["Mod1_ASW"]), max(res_df["Mod2_ASW"]),
              max(res_df["PCA_ASW"]), max(res_df["Our_ASW"]))
    ax.set_ylim(0, max_asw + 0.1 * max_asw)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel('ASW')
    ax.set_xticks(x + width * (len(mod_lbls) - 1) / 2)
    ax.set_xticklabels(data_lbls)
    ax.legend(loc='upper left', ncols=3)

    out_plot = os.path.join(args.save_dir, "Ours_PCA_Mod1_Mod2_ASW.png")
    plt.savefig(out_plot)

    print("Finished!")


if __name__ == "__main__":
    main()