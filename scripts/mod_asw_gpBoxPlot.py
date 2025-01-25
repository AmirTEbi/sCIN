import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.metrics import silhouette_score
import os
import argparse


def compute_asw(embs:np.array, labels:np.array) -> float:
    
    return (silhouette_score(embs, labels) + 1) / 2


def plot_gp_asw_5(df:pd.DataFrame, save_dir:str, legend_names:dict, mod_colors:dict, 
               xticks:None, file_type="png") -> None:
    

    df['prop'] = df['prop'].astype(str)
    prop_order = ["Random", "1", "5", "10", "20", "50", "Paired"]
    df['prop'] = pd.Categorical(df['prop'], categories=prop_order, ordered=True)

    box_width = 0.5  
    group_spacing = 0.5
    num_groups = len(prop_order)
    pos = [
        i * group_spacing for i in range(num_groups)
    ]


    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    sns.boxplot(x="prop", y="asw", hue="mod", data=df, 
                palette=mod_colors, hue_order=["Modality 1", "Modality 2", "Joint"],
                width=box_width, ax=ax)
    ax.set_xticks(pos)
    ax.set_xticklabels(prop_order)
    ax.tick_params(axis='x', pad=10)
    
    legend_handles = [
        Line2D(
            [0], [0], color=mod_colors[modality], linewidth=4, 
            label=legend_names.get(modality, modality)  
        ) for modality in df['mod'].unique()
    ]
    plt.legend(
    handles=legend_handles,
    title="",
    fontsize=14,
    loc='lower center',  
    bbox_to_anchor=(0.5, 0.5),  # 0.5, -0.15
    ncol=3,
    frameon=False
    )
    plt.xlabel("")
    #ax.xaxis.labelpad = 25
    plt.ylabel("ASW", fontsize=12)
    if xticks:
        plt.xticks(xticks['positions'], xticks['labels'], fontsize=12)
    else:
        plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fig5_b_V1.{file_type}"))


def prop_type(value):

    try:
        return(int(value))
    except ValueError:
        if value == "Random":
            return value
        raise argparse.ArgumentTypeError(f"Invalid value for --prop: {value}. Must be an integer or 'Random'.")


def main() -> None:
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--unpaired", action="store_true")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--prop", type=prop_type, nargs="*", 
                        default=[1,5,10,20,50,"Random"], help="In %")
    args = parser.parse_args()

    if args.paired:
        raise NotImplementedError

    if args.unpaired:
        UNPAIRED_MAIN_DIR = f"results/{args.data}/main_unpaired"
        PAIRED_MAIN_DIR = f"results/{args.data}/sCIN/V1"
        REP_DIRS = [f"rep{i+1}" for i in range(0, args.reps)]
        PROP_DIRS = [f"p{p}" for p in args.prop]
        res = pd.DataFrame(columns = ["prop", "rep", "mod", "asw"])

        for i, d in enumerate(REP_DIRS):
            
            # Compute ASW for paired setting in this rep
            paired_rna_embs = np.load(os.path.join(PAIRED_MAIN_DIR,
                                                   f"rep{i+1}",
                                                   "embs",
                                                   f"rna_emb{10*i}.npy"))
            print(paired_rna_embs.shape)
            paired_atac_embs = np.load(os.path.join(PAIRED_MAIN_DIR,
                                                    f"rep{i+1}",
                                                    "embs",
                                                    f"atac_emb{10*i}.npy"))
            print(paired_atac_embs.shape)
            paired_lbls = np.load(os.path.join(PAIRED_MAIN_DIR,
                                               f"rep{i+1}",
                                               "embs",
                                               f"labels_test_{10*i}.npy"))
            print(paired_lbls.shape)
            paired_rna_asw = compute_asw(paired_rna_embs, paired_lbls)
            print(paired_rna_asw)
            paired_atac_asw = compute_asw(paired_atac_embs, paired_lbls)
            paired_joint = np.column_stack((paired_rna_embs,
                                            paired_atac_embs))
            print(paired_atac_asw)
            paired_joint_asw = compute_asw(paired_joint, paired_lbls)
            print(paired_joint_asw)

            paired_rna_row = pd.DataFrame({
                "prop":["Paired"],
                "rep":[i+1],
                "mod":["Modality 1"],
                "asw":[paired_rna_asw]
            })
            paired_atac_row = pd.DataFrame({
                "prop":["Paired"],
                "rep":[i+1],
                "mod":["Modality 2"],
                "asw":[paired_atac_asw]
            })
            paired_joint_row = pd.DataFrame({
                "prop":["Paired"],
                "rep":[i+1],
                "mod":["Joint"],
                "asw":[paired_joint_asw]
            })
            res = pd.concat([res, paired_rna_row, paired_atac_row, 
                             paired_joint_row], ignore_index=True)

            # Compute ASW for unpaired setting in this rep
            rep_dir = os.path.join(UNPAIRED_MAIN_DIR, d)
            for p, p_dir in zip(args.prop, PROP_DIRS):
                prop_dir = os.path.join(rep_dir, p_dir, "embs")
                files = [f for f in os.listdir(prop_dir) \
                         if f.endswith(".npy")]
                
                # print(f"Processing rep_dir: {rep_dir}")
                # print(f"prop_dir: {prop_dir}, files: {files}")

                rna_embs_files = [f for f in files if "rna" in f]
                atac_embs_files = [f for f in files if "atac" in f]
                lbls_files = [f for f in files if "labels" in f]

                for f_r, f_a, lbls in zip(rna_embs_files,
                                          atac_embs_files,
                                          lbls_files):
                    rna_embs = np.load(os.path.join(prop_dir, f_r))
                    atac_embs = np.load(os.path.join(prop_dir, f_a))
                    labels = np.load(os.path.join(prop_dir, lbls))

                    rna_asw = compute_asw(rna_embs, labels)
                    atac_asw = compute_asw(atac_embs, labels)
                    joint = np.column_stack((rna_embs, atac_embs))
                    joint_asw = compute_asw(joint, labels)

                    print(p)
                    rna_row = pd.DataFrame({
                        "prop":[p],
                        "rep":[i+1],
                        "mod":["Modality 1"],
                        "asw":[rna_asw]
                    })
                    atac_row = pd.DataFrame({
                        "prop":[p],
                        "rep":[i+1],
                        "mod":["Modality 2"],
                        "asw":[atac_asw]
                    })
                    joint_row = pd.DataFrame({
                        "prop":[p],
                        "rep":[i+1],
                        "mod":["Joint"],
                        "asw":[joint_asw]
                    })

                    res = pd.concat([res, rna_row, atac_row, joint_row], ignore_index=True)


        print(res['prop'].value_counts())

        PLOT_SAVE_DIR = os.path.join(UNPAIRED_MAIN_DIR, "plots")
        os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
        
        COLORS = {"Modality 1":"#7fc97f", "Modality 2":"#beaed4", "Joint":"#fdc086"}
        LEGENDS = {
            "Modality 1": "RNA",
            "Modality 2": "ATAC",
            "Joint": "Joint"
        }
        XTICKS = {
            "positions": range(7),
            "labels": ["Random", "1%", "5%", "10%", "20%", "50%", "Paired"]
        }
        plot_gp_asw_5(df=res, save_dir=PLOT_SAVE_DIR,legend_names=LEGENDS,
                   mod_colors=COLORS, xticks=XTICKS, file_type="pdf")
                   
        res.to_csv(os.path.join(UNPAIRED_MAIN_DIR, "unpaired_asw_mods_V1.csv"), index=False)

        print("Finish.")


if __name__ == "__main__":
    main()