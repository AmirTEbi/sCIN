from sCIN.assess import assess_joint
from sCIN.utils import (extract_file_extension, 
                        read_embs_from_csv, 
                        read_embs_from_np, 
                        find_matched_files)
import numpy as np 
import pandas as pd
import argparse
from typing import Dict, List, Tuple, Union, Any
import re
import os


MODELS_DIRS_NAMES = [
    "sCIN", "Con-AAE", "MOFA", "Harmony", "scGLUE", "SnapATAC2"
]

def get_rep_from_filename_np(filename: str):
    
    match = re.search(r"\d+", filename)
    if not match:
        return None
    
    found = int(match.group())

    return (found / 10) + 1


def get_rep_from_filename_csv(filename: str):

    match = re.search(r"\d+", filename)
    if not match:
        return None
    
    rep = int(match.group())

    return rep


def get_seed_from_filename_np(filename: str):

    match = re.search(r"\d+", filename)
    if not match:
        return None
    
    found = int(match.group())

    return found


def get_seed_from_filename_csv(filename: str):

    match = re.search(r"\d+", filename)
    if not match:
        return None
    
    found = int(match.group())

    return (found - 1) * 10


def find_files_in_dir(key: str, dir: str) -> List[str]:

    all_files = os.listdir(dir)
    files = [f for f in all_files if key in f]

    return files


def find_all_embs_files(embs_dir: str):

    rna_embs_files = find_files_in_dir(key="rna", dir=embs_dir)
    atac_embs_files = find_files_in_dir(key="atac", dir=embs_dir)
    labels_embs_files = find_files_in_dir(key="label", dir=embs_dir)

    return rna_embs_files, atac_embs_files, labels_embs_files


def get_helpers_for_file_ext_np():

    return read_embs_from_np, get_rep_from_filename_np, get_seed_from_filename_np


def get_helpers_for_file_ext_csv():

    return read_embs_from_csv, get_rep_from_filename_csv, get_seed_from_filename_csv


def compute_joint_metrics_from_embs(res_dir: str,
                                    model_name: str,
                                    data: str,
                                    results: List[Any] = []) -> List[Dict[str, Any]]:

    embs_dir = os.path.join(res_dir, model_name, data, "embs")
    if not os.path.isdir(embs_dir):
        raise FileNotFoundError(f"Embs dir not found for model {model_name}.")

    if not os.listdir(embs_dir):
        raise FileNotFoundError(f"Embs dir is empty for model {model_name}.")

    rna_embs_files, atac_embs_files, labels_embs_files = find_all_embs_files(embs_dir=embs_dir)

    file_ext = extract_file_extension(rna_embs_files[0])
    if file_ext == ".npy":
        read_embs, get_rep, get_seed = get_helpers_for_file_ext_np()
    elif file_ext == ".csv":
        read_embs, get_rep, get_seed = get_helpers_for_file_ext_csv()
    
    for rna_f in rna_embs_files:
        atac_f, lbls_f = find_matched_files(rna_f,
                                            atac_embs_files,
                                            labels_embs_files)
        
        rep = get_rep(rna_f)
        seed = get_seed(rna_f)
        rna_embs, atac_embs, lbls = read_embs([rna_f, atac_f, lbls_f])
        joint_embs = np.column_stack([rna_embs, atac_embs])
        cell_type_acc_joint, _ = assess_joint(joint_embs, lbls, seed)

        results.append({
            "Models": model_name,
            "Replicates": rep,
            "cell_type_acc": cell_type_acc_joint
        })

    return results


def setup_args_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--res_dir", type=str, help="Result directory")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data", type=str)
    parser.add_argument("--models_dirs", nargs='+', type=str, help="List of model directories.")
    parser.add_argument("--all", action="store_true")

    return parser


def main() -> None:

    parser = setup_args_parser()
    args = parser.parse_args()

    model_dirs = args.models_dirs if not args.models_dirs == None else MODELS_DIRS_NAMES
    if args.all:
        res = []
        for model_name in model_dirs:
            res = compute_joint_metrics_from_embs(res_dir=args.res_dir,
                                                  model_name=model_name,
                                                  data=args.data,
                                                  results=res)
        
        res_df = pd.DataFrame(res)
        res_df.to_csv(os.path.join(args.res_dir, f"Joint_metrics_for_all_models_on_{args.data}.csv"))

    else:
        res = []
        res = compute_joint_metrics_from_embs(res_dir=args.res_dir,
                                              model_name=args.model_name,
                                              data=args.data,
                                              results=res)
        
        res_df = pd.DataFrame(res)
        save_dir = os.path.join(args.res_dir, args.model_name, "outs")
        os.makedirs(save_dir, exist_ok=True)
        res_df.to_csv(os.path.join(save_dir, f"Joint_metrics_for_{args.model_name}_on_{args.data}.csv"))


if __name__ == "__main__":
    main()