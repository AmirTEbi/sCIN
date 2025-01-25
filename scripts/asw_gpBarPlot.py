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
from sCIN.utils.utils import load_data
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from scripts.mod_asw_gpBoxPlot import compute_asw
from statistics import fmean
import argparse


def plot_bar_asw(df:pd.DataFrame, save_dir:str, file_type="jpg") -> None:

    labels = df['label'].unique()  
    pivot_df = df.pivot(index='data', columns='label', values='ASW')[labels]

    x = np.arange(len(pivot_df.index))  
    num_labels = len(labels)
    bar_width = 0.15
    space = 0.05

    offsets = np.arange(num_labels) * (bar_width + space)
    offsets -= (offsets[-1] + bar_width) / 2 
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, label in enumerate(labels):
        ax.bar(x + offsets[idx], pivot_df[label], width=bar_width,
               label=label, color=colors[idx])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index)
    ax.set_xlabel("")
    ax.set_ylabel("ASW")
    ax.legend(title="", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    out = os.path.join(save_dir, f"asw_grouped_bars.{file_type}")
    plt.savefig(out, bbox_inches='tight', pad_inches=0)


def seeds_type(value):

    try:
        return(int(value))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value for --seeds: {value}. Must be an integer.")


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--from_file", action="store_true")
    parser.add_argument("--compute", action="store_true")
    parser.add_argument("--all_seeds", action="store_true")
    parser.add_argument("--seeds", type=seeds_type, nargs='*')
    parser.add_argument("--file_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--unpaired", action="store_true")

    args = parser.parse_args()

    FILE_TYPE = "jpg"

    data_paths = {
        "SHARE-seq":("data/share/Ma-2020-RNA.h5ad",
                  "data/share/Ma-2020-ATAC.h5ad"),
        "PBMC":("data/10x/10x-Multiome-Pbmc10k-RNA.h5ad", 
                "data/10x/10x-Multiome-Pbmc10k-ATAC.h5ad"),
        "CITE-seq":("data/cite/rna.h5ad", 
                "data/cite/adt.h5ad")
    }
    paired_embs_paths = {"SHARE-seq":"results/SHARE/sCIN/V1",
                         "PBMC":"results/PBMC/sCIN/V1",
                         "CITE-seq":"results/CITE/sCIN/V1"}
    unpaired_embs_paths = {"SHARE":"results/SHAER/main_unpaired",
                           "PBMC":"results/PBMC/main_unpaired",
                           "CITE":"results/CITE/main_unpaired"}
    if args.all_seeds:
        seeds = np.arange(0, 100, 10).tolist()
    else:
        seeds = args.seeds

    #rep_dirs = [f"rep{int((seed/10) + 1)}" for seed in seeds]

    df = pd.DataFrame(columns=[
        "data", "label", "ASW"
    ])

    if args.compute:
        for data, paths in data_paths.items():
            rna = ad.read_h5ad(paths[0])
            atac = ad.read_h5ad(paths[1])
            rna_counts = rna.layers["norm_raw_counts"]
            atac_counts = atac.layers["norm_raw_counts"]
            print(f"Shape of the data 1: {rna_counts.shape}")
            print(f"Shape of the data2: {atac_counts.shape}")
            print(type(rna.obs["cell_type_encoded"]))
            lbls = rna.obs["cell_type_encoded"].values

            pca = PCA(n_components=256)  # As same as embs

            if args.paired:
                embs_asws_list = []
                mod1_asw_list = []
                mod2_asw_list = []
                pca_asw_list = []
                for seed in seeds:
                    rna_embs = np.load(os.path.join(paired_embs_paths[data], 
                                                    f"rep{int((seed/10)+1)}", 
                                                    "embs", f"rna_emb{seed}.npy"))
                    atac_embs = np.load(os.path.join(paired_embs_paths[data], 
                                                    f"rep{int((seed/10)+1)}", 
                                                    "embs", f"atac_emb{seed}.npy"))
                    embs_lbls = np.load(os.path.join(paired_embs_paths[data], 
                                                    f"rep{int((seed/10)+1)}", 
                                                    "embs", f"labels_test_{seed}.npy"))
                    joint = np.column_stack((rna_embs, atac_embs))
                    embs_asw = compute_asw(joint, embs_lbls)
                    embs_asws_list.append(embs_asw)

                    rna_train, rna_test, atac_train, \
                        atac_test, _,lbls_test = train_test_split(
                        rna_counts, atac_counts, lbls, test_size=0.3, random_state=seed
                    )
                    
                    mod1_asw = compute_asw(rna_test, lbls_test)
                    mod2_asw = compute_asw(atac_test, lbls_test)
                    mod1_asw_list.append(mod1_asw)
                    mod2_asw_list.append(mod2_asw)

                    joint_data_train = np.column_stack((rna_train, atac_train))
                    pca.fit(joint_data_train)
                    joint_data_test = np.column_stack((rna_test, atac_test))
                    joint_data_pca = pca.transform(joint_data_test)
                    joint_data_pca_asw = compute_asw(joint_data_pca, lbls_test)
                    pca_asw_list.append(joint_data_pca_asw)

                avg_embs_asw = fmean(embs_asws_list)
                avg_mod1_asw = fmean(mod1_asw_list)
                avg_mod2_asw = fmean(mod2_asw_list)
                avg_pca_asw = fmean(pca_asw_list)
                temp_dict = {
                    "sCIN":avg_embs_asw,
                    "Modality 1":avg_mod1_asw,
                    "Modality 2":avg_mod2_asw,
                    "PCA":avg_pca_asw
                }
                for lbl, asw in temp_dict.items():
                    row = pd.DataFrame({
                        "data":[data],
                        "label":[lbl],
                        "ASW":asw
                    })
                    df = pd.concat([df, row], ignore_index=True)
                  
            elif args.unpaired:
                raise NotImplementedError

        df.to_csv(os.path.join(args.save_dir, "data_pca_embs_asw.csv"),
                                index=False)
        
        # Plot
        plot_bar_asw(df= df, save_dir=args.save_dir)
            

    elif args.from_file:
        if args.paired:
            df = pd.read_csv(args.file_path)
            plot_bar_asw(df=df, save_dir=args.save_dir)

        elif args.unpaired:
            raise NotImplementedError
          
            
if __name__ == "__main__":
    main()