"""This is a stand-alone script to perform benchmarking when I don't want to run the pipeline."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sc_cool.data_processing.plots import plot_recall_at_k, plot_MedR, plot_cell_type_accuracy
import argparse

def main():

    # Get parameters
    parser = argparse.ArgumentParser(description='Get Parameters')
    parser.add_argument('--measure', type=str, help='Measure')
    parser.add_argument('--data_path', type=str, help='Data Path')
    parser.add_argument('--save_dir', type=str, help='Save Directory')
    parser.add_argument('--model_name', type=str, help='Model Name')
    args = parser.parse_args()

    # Read data
    data = pd.read_csv(args.data_path)
    print(data.head())

    # Make plots
    if args.measure == "recall_at_k":
        plot_recall_at_k(data, save_dir=args.save_dir, model_name=args.model_name)
    elif args.measure == "MedR":
        plot_MedR(data, save_dir=args.save_dir, model_name=args.model_name)    
    elif args.measure == "cell_type_acc":
        plot_cell_type_accuracy(data, save_dir=args.save_dir, model_name=args.model_name)


if __name__ == "__main__":
    main()