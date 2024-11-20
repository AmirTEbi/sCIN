import pandas as pd
import matplotlib.pyplot as plt
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

    grouped_data = data.groupby["Models"]

    plt.figure(figsize=(12, 8))

    for model, group in grouped_data:
        plt.plot(group['k'], group['Recall_at_k'], marker='o', label=model)

    plt.title('Recall@k for Different Models', fontsize=16)
    plt.xlabel('k', fontsize=14)
    plt.ylabel('Recall@k', fontsize=14)
    plt.legend(title='Model', fontsize=12, title_fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir + "\recall_at_k_" + model_name + ".png")


def plot_class_lbl_accuracy(data:pd.DataFrame, save_dir:str, model_name:str) -> None:

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

    raise NotImplementedError


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

    raise NotImplementedError


def main():

    # Get parameters
    parser = argparse.ArgumentParser(description='Get Parameters')
    parser.add_argument('--data_path', type=str, help='Data Path')
    parser.add_argument('--save_dir', type=int, help='Save Directory')
    args = parser.parse_args()

    # Read data
    data = pd.read_csv(args.data_path)
    print(data.head())

    # Make plots
    plot_recall_at_k(data, save_dir=args.save_dir, model_name=args.save_dir)
    



if __name__ == "__main__":
    main()