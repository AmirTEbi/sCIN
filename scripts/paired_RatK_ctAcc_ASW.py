import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import os
import argparse


matplotlib.rcParams["font.family"] = "sans-serif"
# matplotlib.rcParams['font.weight'] = 'bold'

def make_palette(models:list, colors:list) -> dict:

    return {model: color for model, color in zip(models, colors)}


def plot_ratk_24(df: pd.DataFrame, colors: dict, save_dir: str, 
                 file_type="png", legend_names: dict = None, **kwargs) -> None:
    
    """
    Plot Recall@k value for each model at different k neighbor embeddings.

    Parameters
    ----------
    data: pandas.DataFrame
        A data frame containing Recall@k values.

    save_dir: str
        Path to save plots.

    model_name: str
        Name of the model to be include the file name.

    Return
    ----------
    None
    
    """

    grouped_df = df.groupby(['Models', 'k'])['Recall_at_k'].agg(['mean', 
                                                                     'std']).reset_index()

    legend_names = legend_names or {model: model for model in grouped_df["Models"].unique()}

    plt.figure(figsize=(7.7, 6.15))
    ax = plt.gca()
    for model in grouped_df['Models'].unique():
        model_df = grouped_df[grouped_df['Models'] == model]
        plt.errorbar(
            model_df['k'], model_df['mean'], 
            yerr=model_df['std'], fmt='-', 
            label=model, capsize=5, capthick=1, linewidth=2, 
            color=colors[model]
        )
    
    legend_handles = [
        Line2D(
            [0], [0], color=colors[model], linewidth=4, 
            label=legend_names.get(model, model)  
        ) for model in grouped_df['Models'].unique()
    ]
    plt.legend(
        handles=legend_handles, title="", fontsize=18, loc='lower center', 
        bbox_to_anchor=(0.2, 0.62), frameon=False
    )

    plt.xlabel("k", fontsize=20)
    plt.ylabel("Recall@k", fontsize=20)
    xticks_positions = [10, 20, 30, 40, 50]
    plt.xticks(xticks_positions, labels=xticks_positions, fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(0, 0.32)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(save_dir, f"RatK_by_models.{file_type}")
    plt.savefig(out)


def plot_asw_24(df:pd.DataFrame, colors:dict, save_dir:str, file_type="png", 
                xticks=None) -> None:
    
    plt.figure(figsize=(7.2, 6.15))
    ax = plt.gca()
    ax = sns.boxplot(x="Models", y="cell_type_ASW", data=df, palette=colors)

    plt.xlabel("")
    plt.ylabel("ASW", fontsize=20)
    if xticks:
        plt.xticks(xticks['positions'], xticks['labels'], fontsize=18, rotation=45)
    else:
        plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(0, 0.8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(save_dir, f"ASW_by_models.{file_type}")
    plt.savefig(out)


def plot_ct_acc_24(df:pd.DataFrame, colors:dict, save_dir:str, file_type="png", 
                  xticks=None, **kwargs) -> None:

    plt.figure(figsize=(7.1, 5.94))
    ax = plt.gca()

    inverse = kwargs.get("inverse", False)
    if inverse:
        ax = sns.boxplot(x="Models", y="cell_type_acc_2to1", data=df, 
                         palette=colors)
    else:
        ax = sns.boxplot(x="Models", y="cell_type_acc", data=df, 
                         palette=colors)

    plt.xlabel("")
    plt.ylabel("Cell Type Accuracy", fontsize=20)
    if xticks:
        plt.xticks(xticks['positions'], xticks['labels'], fontsize=18, rotation=45)
    else:
        plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(save_dir, f"ctAcc_by_models.{file_type}")
    plt.savefig(out)


def plot_medr_24(df:pd.DataFrame, colors:dict, 
                 save_dir:str, file_type:str="png", 
                 xticks:dict=None) -> None:

    df['norm_med_rank'] = (df["MedR"] - df['MedR'].min()) / \
        (df['MedR'].max() - df['MedR'].min())
    
    plt.figure(figsize=(7.86, 5.94))
    ax = plt.gca()

    sns.boxplot(
        x="Models", y="norm_med_rank", data=df, palette=colors)
    
    plt.xlabel("")
    plt.ylabel("Normalized Median Rank", fontsize=20)
    if xticks:
        plt.xticks(xticks['positions'], xticks['labels'], fontsize=18, rotation=45)
    else:
        plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = os.path.join(save_dir, f"MedR_all.{file_type}")
    plt.savefig(out)


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--metric", type=str)
    args = parser.parse_args()

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#fee08b", "#8c510a"]
    legend = {
        "sCIN":"sCIN",
        "Con-AAE":"Con-AAE",
        "MOFA":"MOFA",
        "Harmony":"Harmony",
        "AE":"Autoencoder",
        "scBridge":"scBridge",
        "SnapATAC2":"SnapATAC2"
    }
    xticks = {
    "positions": [0, 1, 2, 3, 4, 5, 6],
    "labels": ["sCIN", "Con-AAE", "Harmony", "MOFA", "Autoencoder", "scBridge", "SnapATAC2"]
    }

    df = pd.read_csv(args.path)
    models = df["Models"].unique().tolist()
    colors = make_palette(models, colors)
    print(colors)

    if args.all:
        plot_ratk_24(df, colors, args.save_dir, file_type="pdf", legend_names=legend)
        plot_asw_24(df, colors, args.save_dir, file_type="pdf", xticks=xticks)
        plot_ct_acc_24(df, colors, args.save_dir, file_type="pdf", xticks=xticks)
        plot_medr_24(df, colors, args.save_dir, file_type="pdf", xticks=xticks)
    
    else:
        if args.metric == "r_at_k":
            plot_ratk_24(df, colors, args.save_dir, file_type="pdf", legend_names=legend)

        elif args.metric == "ASW":
            plot_asw_24(df, colors, args.save_dir, file_type="png", xticks=xticks)
        
        elif args.metric == "ct_acc":
            plot_ct_acc_24(df, colors, args.save_dir, file_type="png",xticks=xticks)
        
        elif args.metric == "medr":
            plot_medr_24(df, colors, args.save_dir, file_type="png", xticks=xticks)

    print("Finished.")


if __name__ == "__main__":
    main()