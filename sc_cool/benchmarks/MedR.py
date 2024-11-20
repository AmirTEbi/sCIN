import numpy as np 
import os
import pandas as pd
import argparse

def MedR(embs1, embs2):
    """
    Find the median rank of the correct match between two sets of embeddings.

    Parameters
    ----------
    embs1 : numpy.ndarray
        A 2D array of shape (n_samples, n_features) representing the first set of embeddings.
        
    embs2 : numpy.ndarray
        A 2D array of shape (n_samples, n_features) representing the second set of embeddings.

    Returns
    -------
    float
        The median rank of the corresponding embeddings.
    """

    ranklist = []
    num_cells = embs1.shape[0]

    for i, emb1 in enumerate(embs1):
        distlist = []

        for j, emb2 in enumerate(embs2):
            dist = np.linalg.norm(emb1 - emb2)

            if i == j:
                flag = dist

            distlist.append(dist)

        dist_list_sorted = sorted(distlist)
        rank = dist_list_sorted.index(flag)
        ranklist.append(rank)

    assert len(ranklist) == num_cells, "The number of ranks in the ranklist should be equal to the number of cells"

    med_rank = np.median(np.array(ranklist))

    return med_rank


def main():
    """ """

    parser = argparse.ArgumentParser(description="Directory args")
    parser.add_argument("--embs_dir", type=str, help="Path to the embeddings")
    parser.add_argument("--model", type=str, help="name of the model")
    parser.add_argument("--output", type=str, help="output directory")
    args = parser.parse_args()


    emb_files = [f for f in os.listdir(args.embs_dir) if f.endswith(".npy")]
    print(f"Found .npy files: {emb_files}")
    
    emb_files.sort()
    rna_embs_files = [f for f in emb_files if "rna" in f]
    atac_embs_files = [f for f in emb_files if "atac" in f]

    results = []
    for rep_idx, (rna_file, atac_file) in enumerate(zip(rna_embs_files, atac_embs_files), start=1):
        rna_path = os.path.join(args.embs_dir, rna_file)
        atac_path = os.path.join(args.embs_dir, atac_file)

        rna_emb = np.load(rna_path)
        atac_emb = np.load(atac_path)

        print(f"Processing {rna_file} and {atac_file}")

        med_rank = MedR(atac_emb, rna_emb)

        results.append({
            "model":args.model,
            "rep":rep_idx,
            "med_rank":med_rank
        })

    df = pd.DataFrame(results)
    out_file = args.output
    df.to_csv(out_file, index=False)
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()