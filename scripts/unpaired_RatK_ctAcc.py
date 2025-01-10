import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import seaborn as sns
import os
import argparse


def make_palette(models:list, colors:list) -> dict:

    return {model: color for model, color in zip(models, colors)}


def plot_ratk_5(df: pd.DataFrame, colors: dict, save_dir: str, file_type="png", 
                legend_names: dict = None, **kwargs) -> None:
    
    inverse = kwargs.get("inverse", False)
    if inverse:
        grouped_df = df.groupby(["Models", "k"])["Recall_at_k_2to1"].agg(["mean", "std"]).reset_index()
    else:
        grouped_df = df.groupby(["Models", "k"])["Recall_at_k"].agg(["mean", "std"]).reset_index()
    

    legend_names = legend_names or {model: model for model in grouped_df["Models"].unique()}

    plt.figure(figsize=(6, 4))
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
            [0], [0], color=colors[model], linewidth=2, 
            label=legend_names.get(model, model)  
        ) for model in grouped_df['Models'].unique()
    ]
    plt.legend(
        handles=legend_handles, title="", fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5)
    )

    plt.xlabel("k", fontsize=10)
    plt.ylabel("Recall@k", fontsize=10)
    xticks_positions = [10, 20, 30, 40, 50]
    plt.xticks(xticks_positions, labels=xticks_positions, fontsize=10)
    plt.yticks(fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(save_dir, f"Fig5_a.{file_type}")
    plt.savefig(out)


# def plot_asw_5(df:pd.DataFrame, colors:dict, save_dir:str, file_type="png",
#                xticks=None) -> None:

#     plt.figure(figsize=(6, 4))
#     ax = plt.gca()
#     ax = sns.boxplot(x="Models", y="cell_type_ASW", data=df, palette=colors)

#     plt.xlabel("")
#     plt.ylabel("ASW", fontsize=10)
#     if xticks:
#         plt.xticks(xticks['positions'], xticks['labels'], rotation=45, fontsize=10)
#     else:
#         plt.xticks(rotation=45, fontsize=10)
#     plt.yticks(fontsize=10)

#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

#     plt.tight_layout()
#     out = os.path.join(save_dir, f"fig5_b.{file_type}")
#     plt.savefig(out)

    
def plot_ct_acc_5(df:pd.DataFrame, colors:dict, save_dir:str, file_type="png", 
                  xticks=None, **kwargs) -> None:

    plt.figure(figsize=(6, 4))
    ax = plt.gca()

    inverse = kwargs.get("inverse", False)
    if inverse:
        ax = sns.boxplot(x="Models", y="cell_type_acc_2to1", data=df, palette=colors)
    else:
        ax = sns.boxplot(x="Models", y="cell_type_acc", data=df, palette=colors)

    plt.xlabel("")
    plt.ylabel("Cell Type Accuracy", fontsize=10)
    if xticks:
        plt.xticks(xticks['positions'], xticks['labels'], fontsize=10)
    else:
        plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(save_dir, f"fig5_c.{file_type}")
    plt.savefig(out)


def main() -> None:
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--metric", type=str)
    args = parser.parse_args()

    colors = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", 
              "#fb9a99", "#e31a1c", "#fdbf6f"]
    legend = {
        "sCIN_Random":"Random",
        "sCIN_0.01":"1%",
        "sCIN_0.05":"5%",
        "sCIN_0.1":"10%",
        "sCIN_0.2":"20%",
        "sCIN_0.5":"50%",
        "paired":"Paired"
    }

    xticks = {
    "positions": [0, 1, 2, 3, 4, 5, 6],
    "labels": ["Random", "1%", "5%", "10%", "20%", "50%", "Paired"]
    }

    df = pd.read_csv(args.path)
    models = df["Models"].unique().tolist()
    colors = make_palette(models, colors)

    if args.all:
        plot_ratk_5(df, colors, args.save_dir, file_type="pdf", legend_names=legend)
        #plot_asw_8(df, colors, args.save_dir, file_type="pdf", xticks=xticks)
        plot_ct_acc_5(df, colors, args.save_dir, file_type="pdf", xticks=xticks)

    else:
        if args.metric == "r_at_k":
            plot_ratk_5(df, colors, args.save_dir, file_type="pdf", legend_names=legend)

        #elif args.metric == "ASW":
            #plot_asw_5(df, colors, args.save_dir, file_type="pdf", xticks=xticks)
        
        elif args.metric == "ct_acc":
            plot_ct_acc_5(df, colors, args.save_dir, file_type="pdf",xticks=xticks)

    print("Finished.")


if __name__ == "__main__":
    main()