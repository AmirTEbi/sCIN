import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sCIN.models.ConAAE.conAAE import mink
from sCIN.benchmarks.assess import compute_metrics
import os
import argparse


def compute_asw(data:np.array, labels:np.array) -> float:

    return (silhouette_score(data, labels) + 1) / 2


def compute_metrics_paired(embs_dir:str, reps:list, model_name:str, 
                           outfile:str, inv:bool=False) -> None:

    res = pd.DataFrame(columns=["Models", "Replicates",
                                "k", "Recall_at_k",
                                "Recall_at_k_inv",
                                "num_pairs", "num_pairs_inv",
                                "cell_type_acc",
                                "cell_type_acc_inv",
                                "cell_type_ASW"])
    for rep in reps:
        rep_dir = os.path.join(embs_dir, f"rep{rep}/embs")
        rna_files = [f for f in os.listdir(rep_dir) if "rna" in f]
        atac_files = [f for f in os.listdir(rep_dir) if "atac" in f]
        lbls_files = [f for f in os.listdir(rep_dir) if "labels" in f]
        for rna_emb_f, atac_emb_f, lbls_f in zip(rna_files, atac_files, lbls_files):
            print(rna_emb_f)
            print(atac_emb_f)
            print(lbls_f)
            rna_embs = np.load(os.path.join(rep_dir, rna_emb_f))
            atac_embs = np.load(os.path.join(rep_dir, atac_emb_f))
            lbls = np.load(os.path.join(rep_dir, lbls_f))

            ratk, num_pairs, ct_acc = compute_metrics(atac_embs, rna_embs, lbls)
            if inv:
                ratk_inv, num_pairs_inv, ct_acc_inv = compute_metrics(rna_embs,
                                                                      atac_embs,
                                                                      lbls)
            joint = np.column_stack((rna_embs, atac_embs))
            asw = compute_asw(joint, lbls)
            for k, v in ratk.items():
                    v_inv = ratk_inv.get(k, 0)
                    new_row = pd.DataFrame({
                        "Models": [model_name],
                        "Replicates":[rep],
                        "k":[k],
                        "Recall_at_k":[v],
                        "Recall_at_k_inv":[v_inv],
                        "num_pairs": [num_pairs],
                        "num_pairs_inv": [num_pairs_inv],
                        "cell_type_acc": [ct_acc],
                        "cell_type_acc_inv": [ct_acc_inv],
                        "cell_type_ASW": [asw]
                    })
                    res = pd.concat([res, new_row], ignore_index=True)
    
    res.to_csv(os.path.join(embs_dir, f"{outfile}.csv"), index=False)


def compute_metrics_unpaired(embs_dir:str, reps:list, props:list, model_name:str, 
                             outfile:str, inv:bool=False) -> None:
    
    res = pd.DataFrame(columns=["Models", "Replicates", "Mod2_prop",
                                "k", "Recall_at_k",
                                "Recall_at_k_inv",
                                "num_pairs", "num_pairs_inv",
                                "cell_type_acc",
                                "cell_type_acc_inv",
                                "cell_type_ASW"])
    for rep in reps:
        rep_dir = os.path.join(embs_dir, f"rep{rep}")
        for p in props:
            prop_dir = os.path.join(rep_dir, f"p{p}/embs")
            name = f"{model_name}_{p}" if p != "Random" else f"{model_name}_Random"

            rna_files = [f for f in os.listdir(prop_dir) if "rna" in f]
            atac_files = [f for f in os.listdir(prop_dir) if "atac" in f]
            lbls_files = [f for f in os.listdir(prop_dir) if "labels" in f]
            for rna_emb_f, atac_emb_f, lbls_f in zip(rna_files, atac_files, lbls_files):
                print(rna_emb_f)
                print(atac_emb_f)
                print(lbls_f)
                rna_embs = np.load(os.path.join(prop_dir, rna_emb_f))
                atac_embs = np.load(os.path.join(prop_dir, atac_emb_f))
                lbls = np.load(os.path.join(prop_dir, lbls_f))

                ratk, num_pairs, ct_acc = compute_metrics(atac_embs, rna_embs, lbls)
                if inv:
                    ratk_inv, num_pairs_inv, ct_acc_inv = compute_metrics(rna_embs,
                                                                        atac_embs,
                                                                        lbls)
                joint = np.column_stack((rna_embs, atac_embs))
                asw = compute_asw(joint, lbls)
                for k, v in ratk.items():
                    v_inv = ratk_inv.get(k, 0)
                    new_row = pd.DataFrame({
                        "Models": [name],
                        "Replicates":[rep],
                        "Mod2_prop":[p],
                        "k":[k],
                        "Recall_at_k":[v],
                        "Recall_at_k_inv":[v_inv],
                        "num_pairs": [num_pairs],
                        "num_pairs_inv": [num_pairs_inv],
                        "cell_type_acc": [ct_acc],
                        "cell_type_acc_inv": [ct_acc_inv],
                        "cell_type_ASW": [asw]
                    })
                    res = pd.concat([res, new_row], ignore_index=True)

    res.to_csv(os.path.join(embs_dir, f"{outfile}.csv"), index=False)


def seed_values_type(value):

    try:
        return(int(value))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value for --seed_values: {value}. Must be an integer.")


def reps_type(value):

    try:
        return(int(value))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value for --seed_values: {value}. Must be an integer.")


def prop_type(value):

    try:
        return(int(value))
    except ValueError:
        if value == "Random":
            return value
        raise argparse.ArgumentTypeError(f"Invalid value for --prop: {value}. Must be an integer or 'Random'.")
    

def main() -> None:
  
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--paired_embs_dir", type=str)
    parser.add_argument("--unpaired", action="store_true")
    parser.add_argument("--unpaired_embs_dir", type=str)
    parser.add_argument("--paired_unpaired", action="store_true")
    parser.add_argument("--inv", action="store_true") 
    parser.add_argument("--data", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--paired_outfile", type=str)
    parser.add_argument("--unpaired_outfile", type=str)
    # parser.add_argument("--seed_range", action="store_true")
    # parser.add_argument("--seed_start", type=int, default=0)
    # parser.add_argument("--seed_end", type=int, default=100)
    # parser.add_argument("--seed_step", type=int, default=10)
    # parser.add_argument("--seed_values", type=seed_values_type,
    #                     nargs="*")
    parser.add_argument("--reps", type=reps_type, nargs="*", 
                        default=[1,2,3,4,5,6,7,8,9,10])
    parser.add_argument("--prop", type=prop_type, nargs="*", 
                        default=[1,5,10,20,50,"Random"], help="In %")
    args = parser.parse_args()

    # if args.seed_range:
    #     seeds = np.arange(args.seed_start, args.seed_end, args.seed_step).tolist()
    
    # elif args.seed_values:
    #     seeds = args.seed_values
        
    if args.paired:
        compute_metrics_paired(embs_dir=args.paired_embs_dir,
                               reps=args.reps,
                               model_name=args.model,
                               outfile=args.paired_outfile,
                               inv=args.inv)
        

    elif args.unpaired:
        compute_metrics_unpaired(embs_dir=args.unpaired_embs_dir,
                                 reps=args.reps,
                                 props=args.props,
                                 model_name=args.model,
                                 outfile=args.unpaired_outfile,
                                 inv=args.inv)
    
    elif args.paired_unpaired:
        compute_metrics_paired(embs_dir=args.paired_embs_dir,
                               reps=args.reps,
                               model_name=args.model,
                               outfile=args.paired_outfile,
                               inv=args.inv)
        compute_metrics_unpaired(embs_dir=args.unpaired_embs_dir,
                                 reps=args.reps,
                                 props=args.prop,
                                 model_name=args.model,
                                 outfile=args.unpaired_outfile,
                                 inv=args.inv)
   
    print("Finished.")


if __name__ == "__main__":
    main()