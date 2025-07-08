import pandas as pd 
import numpy as np
import anndata as ad
from sCIN.plots import (plot_recall_at_k,
                        plot_asw,
                        plot_cell_type_accuracy,
                        plot_median_rank)
from sCIN.utils import extract_file_extension
from sklearn.model_selection import train_test_split
from configs import plots, model_palette
import matplotlib as mpl
import argparse


def setup_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", help="Name of the plot.")
    parser.add_argument("--save_dir")
    parser.add_argument("--metric_file_path")
    parser.add_argument("--matplot_backend", default="PDF")
    parser.add_argument("--matplot_font",default="sans")
    parser.add_argument("--fig_dpi",default=300)
    parser.add_argument("--savefig_dpi",default=300)
 
    return parser


def main() -> None:

    parser = setup_args()
    args = parser.parse_args()

    mpl.rcParams["backend"] = args.matplot_backend  # Set this to 'AGG' for PNG format. More on https://matplotlib.org/stable/users/explain/figure/backends.html#the-builtin-backends. 
    mpl.rc("font", family=args.matplot_font)
    mpl.rcParams["figure.dpi"] = args.fig_dpi
    mpl.rcParams["savefig.dpi"] = args.savefig_dpi

    metrics_data_frame = pd.read_csv(args.metric_file_path)

    if args.plot == "recall_at_k":
        plot_recall_at_k(metrics_data_frame, 
                         configs=plots, 
                         save_dir=args.save_dir)
    
    elif args.plot == "ASW":
        plot_asw(metrics_data_frame, configs=plots, save_dir=args.save_dir, ax=None)
    elif args.plot == "cell_type_accuracy":
        plot_cell_type_accuracy(metrics_data_frame, configs=plots, save_dir=args.save_dir, ax=None)
    elif args.plot == "median_rank":
        plot_median_rank(metrics_data_frame, configs=plots, save_dir=args.save_dir, ax=None)


if __name__ == "__main__":
    main()