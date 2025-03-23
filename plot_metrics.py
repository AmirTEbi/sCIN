import pandas as pd 
import numpy as np 
from sCIN.plots import (plot_recall_at_k,
                        plot_asw,
                        plot_cell_type_accuracy,
                        plot_median_rank,
                        plot_tsne_original,
                        plot_tsne_embs,
                        plot_all)
from configs import plots
import argparse


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", help="Name of the plot.")
    parser.add_argument("--save_dir")
    parser.add_argument("--metric_file_path")
    args = parser.parse_args()

    metrics_data_frame = pd.read_csv(args.metric_file_path)

    if args.plot == "recall_at_k":
        plot_recall_at_k(metrics_data_frame, 
                         configs=plots, 
                         save_dir=args.save_dir)
    
    elif args.plot == "ASW":
        plot_asw(metrics_data_frame, configs=plots, save_dir=None, ax=None)
    elif args.plot == "cell_type_accuracy":
        plot_cell_type_accuracy(metrics_data_frame, configs=plots, save_dir=None, ax=None)
    elif args.plot == "median_rank":
        plot_median_rank(metrics_data_frame, configs=plots, save_dir=None, ax=None)
    elif args.plot == "tsne_original":

    elif args.plot == "tsne_embs":

    elif args.plot == "all":