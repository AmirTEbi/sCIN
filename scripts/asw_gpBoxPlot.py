import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import argparse


def compute_asw(embs:np.array, labels:np.array) -> float:
    
    return (silhouette_score(embs, labels) + 1) / 2


def plot_gp_asw(df:pd.DataFrame, save_dir:str, color_palette:dict, 
                xticks_order=list, file_type="jpg") -> None:

    df["data"] = pd.Categorical(df["data"])
    box_width = 0.5  
    group_spacing = 1
    num_groups = len(np.unique(df["data"].values))
    pos = [
        i * group_spacing for i in range(num_groups)
    ]

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    sns.boxplot(x="data", y="ASW", hue="label", data=df, 
                palette=color_palette, hue_order=["Modality 1", "Modality 2", "PCA",
                                                  "sCIN"], width=box_width, ax=ax)
    ax.set_xticks(pos)
    ax.set_xticklabels(xticks_order)
    ax.tick_params(axis='x', pad=10, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("ASW", fontsize=14)
    ax.legend(title="", bbox_to_anchor=(0.03, 1), loc='upper left', frameon=False,
              fontsize=14)
    plt.tight_layout()
    out = os.path.join(save_dir, f"asw_grouped_boxes.{file_type}")
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

    FILE_TYPE = "pdf"

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
    unpaired_embs_paths = {"SHARE-seq":"results/SHARE/main_unpaired",
                           "PBMC":"results/PBMC/main_unpaired",
                           "CITE-seq":"results/CITE/main_unpaired"}
    if args.all_seeds:
        seeds = np.arange(0, 100, 10).tolist()
    else:
        seeds = args.seeds

    #rep_dirs = [f"rep{int((seed/10) + 1)}" for seed in seeds]

    df = pd.DataFrame(columns=[
        "data", "rep", "label", "ASW"
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
                    embs_row = pd.DataFrame({
                        "data":[data],
                        "rep":[int((seed/10)+1)],
                        "label":["sCIN"],
                        "ASW":embs_asw
                    })

                    rna_train, rna_test, atac_train, \
                        atac_test, _,lbls_test = train_test_split(
                        rna_counts, atac_counts, lbls, test_size=0.3, random_state=seed
                    )
                    
                    mod1_asw = compute_asw(rna_test, lbls_test)
                    mod2_asw = compute_asw(atac_test, lbls_test)
                    # mod1_asw_list.append(mod1_asw)
                    # mod2_asw_list.append(mod2_asw)
                    mod1_row = pd.DataFrame({
                        "data":[data],
                        "rep":[int((seed/10)+1)],
                        "label":["Modality 1"],
                        "ASW":mod1_asw
                    })
                    mod2_row = pd.DataFrame({
                        "data":[data],
                        "rep":[int((seed/10)+1)],
                        "label":["Modality 2"],
                        "ASW":mod2_asw
                    })

                    joint_data_train = np.column_stack((rna_train, atac_train))
                    pca.fit(joint_data_train)
                    joint_data_test = np.column_stack((rna_test, atac_test))
                    joint_data_pca = pca.transform(joint_data_test)
                    joint_data_pca_asw = compute_asw(joint_data_pca, lbls_test)
                    pca_row = pd.DataFrame({
                        "data":[data],
                        "rep":[int((seed/10)+1)],
                        "label":["PCA"],
                        "ASW":joint_data_pca_asw
                    })

                    df = pd.concat([df, embs_row, mod1_row, mod2_row, pca_row],
                                   ignore_index=True)
                  
            elif args.unpaired:
                raise NotImplementedError

        df.to_csv(os.path.join(args.save_dir, "data_pca_embs_asw.csv"),
                                index=False)
        
        # Plot
        color_palette = {
            "Modality 1":"#d7191c",
            "Modality 2":"#fdae61",
            "PCA":"#abdda4",
            "sCIN":"#2b83ba"
        }
        xticks_order = ["CITE-seq", "SHARE-seq", "PBMC"]
        plot_gp_asw(df= df, save_dir=args.save_dir, color_palette=color_palette,
                    xticks_order=xticks_order, file_type=FILE_TYPE)
            

    elif args.from_file:
        if args.paired:
            df = pd.read_csv(args.file_path)
            color_palette = {
            "Modality 1":"#d7191c",
            "Modality 2":"#fdae61",
            "PCA":"#abdda4",
            "sCIN":"#2b83ba"
            }
            xticks_order = ["CITE-seq", "SHARE-seq", "PBMC"]
            plot_gp_asw(df= df, save_dir=args.save_dir, color_palette=color_palette,
                        xticks_order=xticks_order, file_type=FILE_TYPE)

        elif args.unpaired:
            raise NotImplementedError
          
            
if __name__ == "__main__":
    main()