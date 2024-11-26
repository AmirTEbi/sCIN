"""This is a stand-alone script to perform benchmarking when I don't want to run the pipeline."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sc_cool.data_processing.plots import plot_recall_at_k, plot_MedR, plot_cell_type_accuracy, plot_asw
from sc_cool.benchmarks.assess import compute_metrics
import argparse
import os

def main():

    # Get parameters
    parser = argparse.ArgumentParser(description='Get Parameters')
    parser.add_argument('input_dir', type=str, help='Data Path')
    parser.add_argument('save_dir', type=str, help='Save Directory')
    parser.add_argument('model_name', type=str, help='Model Name')
    parser.add_argument('--metric', type=str, help='Metric')
    parser.add_argument('--metric_file', type=str, help='Metric file')
    parser.add_argument('--compute', action='store_true', help='Compute metrics')
    parser.add_argument('--reverse', action='store_true', help='Compute metrics from RNA to ATAC')
    parser.add_argument('--plot', action='store_true', help='Plot the specified measure')
    args = parser.parse_args()

    # Compute metrics if needed
    if args.compute:

        files = [f for f in os.listdir(args.input_dir) if f.endswith(".npy")]
        print(f"Found .npy files: {files}")
        
        files.sort()
        rna_embs_files = [f for f in files if "rna" in f]
        atac_embs_files = [f for f in files if "atac" in f]
        lbls_files = [f for f in files if "labels" in f]

        results = []
        for rep_idx, (rna_file, atac_file, lbl_file) in enumerate(zip(rna_embs_files, atac_embs_files, lbls_files), start=1):

            rna_path = os.path.join(args.input_dir, rna_file)
            atac_path = os.path.join(args.input_dir, atac_file)
            lbls_path = os.path.join(args.input_dir, lbl_file)

            rna_emb = np.load(rna_path)
            atac_emb = np.load(atac_path)
            lbls = np.load(lbls_path)

            print(f"Processing {rna_file} and {atac_file} and {lbl_file}")

            
            if args.reverse:
                recall_at_k, number_of_pairs, ct_acc = compute_metrics(mod1_embs=rna_emb, mod2_embs=atac_emb, labels=lbls)
            
            else:
                recall_at_k, number_of_pairs, ct_acc = compute_metrics(mod1_embs=atac_emb, mod2_embs=rna_emb, labels=lbls)


            results.append({
            "Models":args.model_name,
            "Replicates":rep_idx,
            "Recall_at_k":recall_at_k,
            "Num_pairs":number_of_pairs,
            "Class_label_acc":ct_acc
            })

            #break

        df = pd.DataFrame(results)
        out_file = f"Metrics_{args.model_name}.csv"
        df.to_csv(args.save_dir + "/" + out_file, index=False)
        print(f"Results saved to {args.save_dir}" + "/" + out_file)
                
    
    # Make plots if needed
    if args.plot:

        # Read data
        data_path = args.input_dir + "\\" + args.metric_file
        data = pd.read_csv(data_path)
        print(data.head())  

        if args.metric == "recall_at_k":
            plot_recall_at_k(data, save_dir=args.save_dir, model_name=args.model_name)
        elif args.metric == "MedR":
            plot_MedR(data, save_dir=args.save_dir, model_name=args.model_name)    
        elif args.metric == "cell_type_acc":
            plot_cell_type_accuracy(data, save_dir=args.save_dir, model_name=args.model_name)
        elif args.metric == "ASW":
            plot_asw(data, save_dir=args.save_dir, model_name=args.model_name)

    print("Finished.")


if __name__ == "__main__":
    main()