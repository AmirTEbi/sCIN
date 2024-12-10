"""
This is a stand-alone script to perform benchmarking when I don't want to run the pipeline.

Tutorial:

> cd sc-cool

# Plot a specific metric based on a pre-existing evaluation file 
sc-cool > python -m sc_cool.benchmarks.benchmarks --save_dir "..." model_name "all" --metric "ASW" --metric_path "..." --plot

# Plot Recall@k, Cell type accuracy, and ASW metrics based on a pre-existing evaluation file
sc-cool > python -m sc_cool.benchmarks.benchmarks --save_dir "..." model_name "all" --metric "ASW" --metric_path "..." --plot_all

# Compute all integration metrics based on pre-existing embeddings
sc-cool > python -m sc_cool.benchmarks.benchmarks --input_emb_dir "..." --save_dir "..." --model_name "scCOOL" --compute 

# Compute all integration metrics for reverse modality alignment setting. Just add a --reverse flag :) !
sc-cool > python -m sc_cool.benchmarks.benchmarks --input_emb_dir "..." --save_dir "..." --model_name "scCOOL" --compute --reverse

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sc_cool.data_processing.plots import plot_recall_at_k, plot_MedR, plot_cell_type_accuracy, plot_asw
from sc_cool.benchmarks.assess import compute_metrics
import argparse
import os

def main():

    # Get options
    parser = argparse.ArgumentParser(description='Get Options')
    parser.add_argument('--input_emb_dir', type=str, help='Input directory for embeddings')
    parser.add_argument('--save_dir', type=str, help='Save Directory')
    parser.add_argument('--model_name', type=str, help='Model Name', default="all")
    parser.add_argument('--metric', type=str, help='Metric to be computed')
    parser.add_argument('--metric_path', type=str, help='Path to a metric file')
    parser.add_argument('--compute', action='store_true', help='Compute metrics')
    parser.add_argument('--reverse', action='store_true', help='Compute metrics in reverse setting')
    parser.add_argument('--plot', action='store_true', help='Plot the specified metric')
    parser.add_argument('--plot_all', action='store_true', help='Plot recall_at_k, cell_type_acc, ASW metrics')
    args = parser.parse_args()

    # Compute metrics if needed
    if args.compute:

        files = [f for f in os.listdir(args.input_emb_dir) if f.endswith(".npy")]
        print(f"Found .npy files: {files}")
        
        files.sort()
        rna_embs_files = [f for f in files if "rna" in f]
        atac_embs_files = [f for f in files if "atac" in f]
        lbls_files = [f for f in files if "labels" in f]

        results = []
        for rep_idx, (rna_file, atac_file, lbl_file) in enumerate(zip(rna_embs_files, atac_embs_files, lbls_files), start=1):

            rna_path = os.path.join(args.input_emb_dir, rna_file)
            atac_path = os.path.join(args.input_emb_dir, atac_file)
            lbls_path = os.path.join(args.input_emb_dir, lbl_file)

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
            "num_pairs":number_of_pairs,
            "cell_type_acc":ct_acc
            })

            #break

        df = pd.DataFrame(results)
        out_file = f"Metrics_{args.model_name}.csv"
        df.to_csv(args.save_dir + "/" + out_file, index=False)
        print(f"Results saved to {args.save_dir}" + "/" + out_file)
                
    
    # Make plots if needed
    if args.plot:

        # Read data
        data = pd.read_csv(args.metric_path)
        print(data.head())  

        if args.metric == "recall_at_k":
            plot_recall_at_k(data, save_dir=args.save_dir, model_name=args.model_name)
        elif args.metric == "MedR":
            plot_MedR(data, save_dir=args.save_dir, model_name=args.model_name)    
        elif args.metric == "cell_type_acc":
            plot_cell_type_accuracy(data, save_dir=args.save_dir, model_name=args.model_name)
        elif args.metric == "ASW":
            plot_asw(data, save_dir=args.save_dir, model_name=args.model_name)

    elif args.plot_all:
        plot_recall_at_k(data, save_dir=args.save_dir, model_name=args.model_name)
        plot_cell_type_accuracy(data, save_dir=args.save_dir, model_name=args.model_name)
        plot_asw(data, save_dir=args.save_dir, model_name=args.model_name)

    print("Finished.")


if __name__ == "__main__":
    main()