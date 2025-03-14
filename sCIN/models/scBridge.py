import pandas as pd
import numpy as np
import importlib.metadata
import pyranges as pr
import anndata as ad
from scipy.sparse import csr_matrix

def extend_to_promoter(gtf_df: pd.DataFrame) -> pd.DataFrame:
    """Extend genomic regions to promoters."""
    plus_strand = gtf_df["strand"] == "+"
    
    gtf_df["Start"] = np.where(plus_strand, np.maximum(0, gtf_df["Start"] - 2000), gtf_df["End"])
    gtf_df["End"] = np.where(plus_strand, gtf_df["Start"], gtf_df["End"] + 2000)
    
    return gtf_df[["Chromosome", "Start", "End", "gene_name"]]


def compute_gene_activity(atac: ad.AnnData, gtf_path: str) -> (np.ndarray, list):
    """Compute gene activity matrix by mapping ATAC peaks to gene body + promoter."""

    std_chrs = {f'chr{i}' for i in range(1, 23)} | {"chrX", "chrY", "chrM"}

    peaks_df = atac.var.reset_index()

    rename_dict = {
        "index": "peak_id",
        "chrom": "Chromosome",
        "chromStart": "Start",
        "chromEnd": "End"
    }
    peaks_df.rename(columns={k: v for k, v in rename_dict.items() if k in peaks_df.columns}, inplace=True)

    if "peak_id" not in peaks_df.columns:
        peaks_df["peak_id"] = peaks_df.index.astype(str)

    if "Chromosome" in peaks_df.columns:
        peaks_df = peaks_df.loc[peaks_df["Chromosome"].isin(std_chrs), ["peak_id", "Chromosome", "Start", "End"]].copy()
    else:
        raise KeyError("Chromosome column is missing in peaks_df")

    peaks_ranges = pr.PyRanges(peaks_df)

    gtf_df = pd.read_csv(
        gtf_path, sep="\t", comment="#", header=None,
        names=["Chromosome", "source", "feature", "Start", "End", "score", "strand", "frame", "attribute"],
        dtype={"Chromosome": str}  # Ensure Chromosome is string
    )
    gtf_df["Chromosome"] = np.where(
        gtf_df["Chromosome"].str.startswith("chr"), gtf_df["Chromosome"], "chr" + gtf_df["Chromosome"]
    )
    gtf_df = gtf_df.loc[(gtf_df["feature"] == "gene") & (gtf_df["Chromosome"].isin(std_chrs))].copy()
    gtf_df["gene_name"] = gtf_df["attribute"].str.extract(r'gene_name "([^"]+)"')

    gtf_pr_df = extend_to_promoter(gtf_df)

    gtf_ranges = pr.PyRanges(gtf_pr_df)

    overlap_df = peaks_ranges.join(gtf_ranges, how="left").df

    peaks_df.set_index(["Chromosome", "Start", "End"], inplace=True)
    overlap_df["peak_id"] = peaks_df.loc[
        overlap_df.set_index(["Chromosome", "Start", "End"]).index, "peak_id"
    ].values

    gene2peak = overlap_df.groupby("gene_name")["peak_id"].apply(list)
    print(f"gene2peak: {gene2peak}")

    num_cells = atac.n_obs
    genes = list(gene2peak.keys())
    gene_activity = np.zeros((num_cells, len(genes)))

    for i, gene in enumerate(genes):
        peak_indices = gene2peak[gene]

        peak_indices = [int(idx) for idx in peak_indices if str(idx).isdigit() and int(idx) < atac.shape[1]]

        if not peak_indices:
            continue  

        peak_names = atac.var_names[peak_indices]

        gene_activity[:, i] = atac[:, peak_names].X.sum(axis=1).A.ravel()

    return gene_activity, genes


def compute_tfidf(gene_act_mat: np.array) -> np.array:
    """Compute TF-IDF normalization for gene activity matrix."""
    tf = gene_act_mat / np.maximum(gene_act_mat.sum(axis=1, keepdims=True), 1e-9)
    idf = np.log1p(gene_act_mat.shape[0] / (1 + (gene_act_mat > 0).sum(axis=0)))

    return tf * idf

def center_data(array: np.array) -> np.array:
    """Standardize data using z-score normalization."""
    mean, std = np.mean(array, axis=0), np.std(array, axis=0)
    
    return (array - mean) / np.maximum(std, 1e-9)  


def preprocess(rna: ad.AnnData, atac: ad.AnnData, gtf_file: str) -> (ad.AnnData, ad.AnnData):
    """Preprocess ATAC and RNA data for integration ensuring shared features."""
    
    gene_act_mat, genes = compute_gene_activity(atac, gtf_file)
    print("Gene activity computed.")

    norm_gam = compute_tfidf(gene_act_mat)
    print("TF-IDF normalization done.")

    scaled_norm_gam = center_data(norm_gam)
    print("Z-score normalization done.")

    new_atac = ad.AnnData(X=csr_matrix(gene_act_mat), var=pd.DataFrame(index=genes))
    if "cell_type" in atac.obs.columns:
        new_atac.obs["CellType"] = atac.obs["cell_type"]
    else:
        print("Warning: 'cell_type' column missing in ATAC metadata.")

    if "gene_name" in rna.var.columns:
        rna.var = rna.var.set_index("gene_name", drop=False)
    else:
        raise KeyError("The RNA AnnData object must have a 'gene_name' column in rna.var.")

    if not rna.var.index.is_unique:
        print("Warning: RNA gene names are not unique. Removing duplicates.")
        rna = rna[:, ~rna.var.index.duplicated()].copy()

    shared_genes = set(rna.var.index).intersection(genes)
    # print(f"Total RNA genes (after setting index): {len(rna.var.index)}")
    # print(f"Total ATAC genes (from gene activity): {len(genes)}")
    # print(f"Shared genes: {len(shared_genes)}")

    shared_genes_sorted = sorted(shared_genes)
    filtered_rna = rna[:, shared_genes_sorted].copy()
    filtered_atac = new_atac[:, shared_genes_sorted].copy()

    # Remove duplicate 'gene_name' column
    if "gene_name" in filtered_rna.var.columns and filtered_rna.var.index.name == "gene_name":
        filtered_rna.var = filtered_rna.var.drop(columns=["gene_name"])
    if "gene_name" in filtered_atac.var.columns and filtered_atac.var.index.name == "gene_name":
        filtered_atac.var = filtered_atac.var.drop(columns=["gene_name"])
    
    return filtered_rna, filtered_atac