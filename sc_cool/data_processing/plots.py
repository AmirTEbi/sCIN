"""A module to draw plots."""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns


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
        "Ours": "#e41a1c",  
        "Con-AAE": "#377eb8", 
        "Harmony": "#4daf4a",  
        "MOFA": "#984ea3",  
        "AE": "#ff7f00", 
    }


    grouped_data = data.groupby(['Models', 'k'])['Recall_at_k'].agg(['mean', 'std']).reset_index()

    plt.figure(figsize=(6, 4))
    ax = plt.gca()

    for model in grouped_data['Models'].unique():
        
        model_data = grouped_data[grouped_data['Models'] == model]
        plt.errorbar(
            model_data['k'], model_data['mean'], yerr=model_data['std'],
            fmt='-', label=model, capsize=5, capthick=1, linewidth=2,
            color=model_colors[model]
        )

    #plt.title('Recall@k for Different Models', fontsize=16)
    legend_handles = [
        Line2D(
            [0], [0], color=model_colors[model], linewidth=2, label=model
        ) for model in model_colors
    ]
    plt.legend(
        handles=legend_handles, title='Model', fontsize=12, title_fontsize=12, 
        loc='center left', bbox_to_anchor=(1, 0.5)
    )

    plt.xlabel("k", fontsize=12)
    plt.ylabel("Recall@k", fontsize=12)
    plt.xticks([10, 20, 30, 40, 50], fontsize=12)
    plt.yticks(fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

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
        "Ours": "#e41a1c",  
        "Con-AAE": "#377eb8", 
        "Harmony": "#4daf4a",  
        "MOFA": "#984ea3",  
        "AE": "#ff7f00", 
    }

    model_colors = data["Models"].map(colors)

    #sns.set_theme(style="whitegrid")
    plt.figure(figsize=(4, 4))
    ax = plt.gca()

    ax = sns.boxplot(x="Models", y="Class_label_acc", data=data, palette=colors)

    #plt.title("Models' Performances in Cell Type Accuracy", fontsize=16)
    plt.xlabel("")
    plt.ylabel("Cell Type Accuracy", fontsize=12)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
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
        "Ours": "#e41a1c",  
        "Con-AAE": "#377eb8", 
        "Harmony": "#4daf4a",  
        "MOFA": "#984ea3",  
        "AE": "#ff7f00", 
    }

    model_colors = data["model"].map(colors)

    #sns.set_theme(style="whitegrid")
    plt.figure(figsize=(4, 4))
    ax = plt.gca()

    # Boxplot
    sns.boxplot(
        x="model", y="MedR", data=data, palette=colors)

    # Add points around the boxes
    #sns.stripplot(
        #x="model", y="MedR", data=data, 
        #jitter=True,  # Adds jitter for better visibility
        #color="black",  # Color of the points
        #alpha=0.6,  # Transparency
        #size=4  # Point size
    #)

    #plt.title("Model Performances in Median Rank (MedR)", fontsize=16)
    plt.xlabel("")
    plt.ylabel("Median Rank (MedR)", fontsize=12)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir + "\\MedR_" + model_name + ".png")


def plot_asw(data:pd.DataFrame, save_dir:str, model_name:str) -> None:

    """
    Plot Average Silhouette Width (ASW) value for each model.

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
        "Ours": "#e41a1c",  
        "Con-AAE": "#377eb8", 
        "Harmony": "#4daf4a",  
        "MOFA": "#984ea3",  
        "AE": "#ff7f00", 
    }

    model_colors = data["Models"].map(colors)

    plt.figure(figsize=(4, 4))
    ax = plt.gca()

    ax = sns.boxplot(x="Models", y="Cell_type_ASW", data=data, palette=colors)

    #plt.title("Models' Performances in Cell Type Clustering", fontsize=16)
    plt.xlabel("")
    plt.ylabel("ASW", fontsize=12)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir + "\\asw_" + model_name + ".png")