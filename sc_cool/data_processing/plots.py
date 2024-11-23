"""A module to draw plots."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def plot_recall_at_k(data:pd.DataFrame, save_dir:str, model_name:str) -> None:

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

    model_colors = {
        "Ours": "#1f77b4",  
        "Con-AAE": "#ff7f0e", 
        "Harmony": "#2ca02c",  
        "MOFA": "#d62728",  
        "AE": "#9467bd", 
    }


    grouped_data = data.groupby(['Models', 'k'])['Recall_at_k'].agg(['mean', 'std']).reset_index()

    plt.figure(figsize=(12, 8))

    for model in grouped_data['Models'].unique():
        
        model_data = grouped_data[grouped_data['Models'] == model]
        plt.errorbar(
            model_data['k'], model_data['mean'], yerr=model_data['std'],
            fmt='-o', label=model, capsize=5, capthick=1, linewidth=2,
            color=model_colors[model]
        )

    plt.title('Recall@k for Different Models', fontsize=16)
    plt.xlabel('k', fontsize=14)
    plt.ylabel('Recall@k', fontsize=14)
    plt.legend(title='Model', fontsize=12, title_fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks([10, 20, 30, 40, 50], fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir + "\\recall_at_k_" + model_name + ".png")


def plot_cell_type_accuracy(data:pd.DataFrame, save_dir:str, model_name:str) -> None:

    """
    Plot Class Label Accuracy for each model.

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

    colors = {
        "Ours": "#1f77b4",  
        "Con-AAE": "#ff7f0e", 
        "Harmony": "#2ca02c",  
        "MOFA": "#d62728",  
        "AE": "#9467bd", 
    }

    model_colors = data["Models"].map(colors)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    ax = sns.boxplot(x="Models", y="Class_label_acc", data=data, palette=colors, showmeans=True, meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black"})

    plt.title("Models' Performances in Cell Type Accuracy", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Cell Type Accuracy", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir + "\\cell_type_acc_" + model_name + ".png")
    
    


def plot_MedR(data:pd.DataFrame, save_dir:str, model_name:str) -> None:

    """
    Plot Median Rank value for each model.

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

    colors = {
        "Ours": "#1f77b4",  
        "ConAAE": "#ff7f0e", 
        "Harmony": "#2ca02c",  
        "MOFA": "#d62728",  
        "AE": "#9467bd", 
    }

    model_colors = data["model"].map(colors)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    ax = sns.boxplot(x="model", y="MedR", data=data, palette=colors, showmeans=True, meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black"})

    plt.title("Model Performances in Median Rank (MedR)", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Median Rank (MedR)", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir + "\\MedR_" + model_name + ".png")