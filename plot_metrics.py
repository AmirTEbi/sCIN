import pandas as pd 
import numpy as np
import anndata as ad
from sCIN.plots import (plot_recall_at_k,
                        plot_asw,
                        plot_cell_type_accuracy,
                        plot_median_rank,
                        compute_tsne_original,
                        compute_tsne_embs,
                        plot_tsne_original,
                        plot_tsne_embs,
                        plot_all)
from sCIN.utils import extract_file_extension
from sklearn.model_selection import train_test_split
from configs import plots
import argparse


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", help="Name of the plot.")
    parser.add_argument("--save_dir")
    parser.add_argument("--metric_file_path")
    parser.add_argument("--compute_tsne", action="store_true")
    parser.add_argument("--plot_tsne", action="store_true")
    parser.add_argument("--mod1_anndata_file")
    parser.add_argument("--mod2_anndata_file")
    parser.add_argument("--mod1_embs_file")
    parser.add_argument("--mod2_embs_file")
    parser.add_argument("--embs_labels_file")
    parser.add_argument("--tsne_original_reps_file")
    parser.add_argument("--original_labels_file")
    parser.add_argument("--tsne_embs_reps_file")
    args = parser.parse_args()

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
    elif args.plot == "tsne_original":
        if args.compute_tsne:
            cfg = plots["tSNE_original"]
            mod1_anndata = ad.read_h5ad(args.mod1_anndata_file)
            mod2_anndata = ad.read_h5ad(args.mod2_anndata_file)
            tsne_original_reps, labels_original_test = compute_tsne_original(mod1_anndata,
                                                                             mod2_anndata,
                                                                             num_components=cfg["num_components"],
                                                                             test_size=cfg["test_size"],
                                                                             init=cfg["init_method"],
                                                                             learning_rate=cfg["learning_rate"])
            plot_tsne_original(tsne_original_reps, configs=cfg,
                               original_labels=labels_original_test,
                               save_dir=args.save_dir)
        
        elif args.plot_tsne:
            file_ext = extract_file_extension(args.tsne_original_reps_file)
            if file_ext == ".npy":
                tsne_original_reps = np.load(args.tsne_original_reps_file)
                labels_original = np.load(args.original_labels_file)
                
            elif file_ext == ".csv":
                tsne_original_reps = pd.read_csv(args.tsne_original_reps_file)
                labels_original_df = pd.read_csv(args.original_labels_file)
                labels_original = labels_original_df.values

            plot_tsne_original(tsne_original_reps, configs=cfg, 
                               seed=0, original_labels=labels_original,
                               save_dir=args.save_dir)

    elif args.plot == "tsne_embs":
        if args.compute_tsne: 
            cfg = plots["tSNE_embs"]
            file_ext = extract_file_extension(args.mod1_embs_file)
            if file_ext == ".npy":
                mod1_embs = np.load(args.mod1_embs_file)
                mod2_embs = np.load(args.mod2_embs_file)
                embs_labels = np.load(args.embs_labels_file)
                    
            elif file_ext == ".csv":
                mod1_embs_df = pd.read_csv(args.mod1_embs_file)
                mod2_embs_df = pd.read_csv(args.mod2_embs_file)
                embs_labels_df = pd.read_csv(args.embs_labels_file)
                mod1_embs = mod1_embs_df.values
                mod2_embs = mod2_embs_df.values
                embs_labels = embs_labels_df.values

            tsne_embs_reps = compute_tsne_embs(mod1_embs, mod2_embs, 
                                               num_components=cfg["num_components"],
                                               init=cfg["init_method"], 
                                               learning_rate=cfg["learning_rate"])
            
            plot_tsne_embs(tsne_embs_reps, labels=embs_labels, configs=cfg, save_dir=args.save_dir)

        elif args.plot_tsne:
            file_ext = extract_file_extension(args.tsne_embs_reps_file)
            if file_ext == ".npy":
                tsne_embs_reps = np.load(args.tsne_embs_reps_file)
                embs_labels = np.load(args.embs_labels_file)
            
            elif file_ext == ".csv":
                tsne_embs_reps_df = pd.read_csv(args.tsne_embs_reps_file)
                embs_labels_df = pd.read_csv(args.embs_labels_file)
                tsne_embs_reps = tsne_embs_reps_df.values
                embs_labels = embs_labels_df.values
            
            plot_tsne_embs(tsne_embs_reps, labels=embs_labels, configs=cfg, save_dir=args.save_dir)

    elif args.plot == "all":
        plot_all(
            data_frame=metrics_data_frame,
            save_dir=args.save_dir,
            configs=plots,
            plot_tsne=True,
            tsne_reps_original_file=args.tsne_original_reps_file,
            tsne_reps_embs_file=args.tsne_embs_reps_file,
            labels_original_file=None,
            compute_tsne=False,
            mod1_anndata_file=args.mod1_anndata_file,
            mod2_anndata_file=args.mod2_anndata_file,
            mod1_embs_file=args.mod1_embs_file,
            mod2_embs_file=args.mod2_embs_file,
            labels_embs_file=args.embs_labels_file,
        )


if __name__ == "__main__":
    main()