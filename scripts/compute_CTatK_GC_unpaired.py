from sCIN.assess import compute_graph_connectivity, cell_type_at_k_unpaired
import numpy as np 
import pandas as pd 
import argparse
import os 

def setup_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)

    return parser


def main():

    parser = setup_args()
    args = parser.parse_args()

    





if __name__ == "__main__":
    main()