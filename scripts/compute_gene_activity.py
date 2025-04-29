import pandas as pd 
import numpy as np
from joblib import Parallel, delayed
import anndata as ad 
import pyranges as pr 
from typing import *
import argparse
import os

def compute_gene_activity(atac_anndata: ad.AnnData,
                          rna_anndata: ad.AnnData,
                          gtf_df: pd.DataFrame,
                          feature: str = "gene",
                          save_in_anndata: bool = False) -> Union[Tuple[np.ndarray, ad.AnnData], np.ndarray]:

    gtf_df = gtf_df[gtf_df["Feature"] == feature].copy()

    if "gene_name" in rna_anndata.var.columns and "gene_name" in gtf_df.columns:
        id_type = "gene_name"
    elif "gene_id" in rna_anndata.var.columns and "gene_id" in gtf_df.columns:
        id_type = "gene_id"
    else:
        raise KeyError("To match features, both RNA var and GTF must share either 'gene_id' or 'gene_name'.")

    rna_var = rna_anndata.var.copy()
    rna_var = rna_var[rna_var[id_type].notnull()]
    gtf_df = gtf_df[gtf_df[id_type].notnull()].copy()

    rna_ids = (
        rna_var[id_type]
        .str.upper()
        .str.replace(r'\..*', '', regex=True)
    )
    gtf_ids = (
        gtf_df[id_type]
        .str.upper()
        .str.replace(r'\..*', '', regex=True)
    )

    print(f"Using identifier '{id_type}' for matching")
    print("RNA identifiers sample:")
    print(rna_ids.head())
    print("GTF identifiers sample:")
    print(gtf_ids.head())

    shared_genes = np.intersect1d(rna_ids.values, gtf_ids.values)
    if len(shared_genes) == 0:
        raise ValueError("No shared identifiers found between RNA and GTF. Check formatting.")
    print(f"Number of shared {id_type}s: {len(shared_genes)}")

    gtf_df[id_type] = gtf_ids.values
    gtf_df = gtf_df[gtf_df[id_type].isin(shared_genes)].copy()

    gtf_df = gtf_df[["Chromosome", "Start", "End", "Strand", id_type]].copy()
    gtf_df.columns = ["Chromosome", "Start", "End", "Strand", "Gene"]

    gtf_df["Promoter_Start"] = gtf_df.apply(
        lambda r: max(0, r.Start - 2000) if r.Strand == "+" else max(0, r.End - 2000), axis=1
    )
    gtf_df["Promoter_End"] = gtf_df.apply(
        lambda r: r.Start + 2000 if r.Strand == "+" else r.End + 2000, axis=1
    )

    if not gtf_df.Chromosome.str.startswith('chr').all():
        gtf_df["Chromosome"] = "chr" + gtf_df["Chromosome"].astype(str)

    gene_ranges = pr.PyRanges(pd.DataFrame({
        "Chromosome": gtf_df["Chromosome"],
        "Start": gtf_df["Promoter_Start"],
        "End": gtf_df["Promoter_End"],
        "Gene": gtf_df["Gene"],
    }))
    peak_df = pd.DataFrame({'peak': atac_anndata.var_names})
    peak_df = peak_df['peak'].str.extract(r"(?P<chr>[^:]+):(?P<start>\d+)-(?P<end>\d+)")
    peak_df[['start','end']] = peak_df[['start','end']].astype(int)
    atac_ranges = pr.PyRanges(chromosomes=peak_df['chr'], starts=peak_df['start'], ends=peak_df['end'])

    overlaps = atac_ranges.join(gene_ranges)
    if overlaps.df.empty:
        raise ValueError("No overlaps found between peaks and promoter regions.")

    overlaps_df = overlaps.df.copy()
    overlaps_df['peak_key'] = (
        overlaps_df.Chromosome + ':' + overlaps_df.Start.astype(str) + '-' + overlaps_df.End.astype(str)
    )
    overlaps_df['Gene'] = overlaps_df.Gene

    peak_to_idx = {
        f"{ch}:{s}-{e}": idx
        for idx, (ch, s, e) in enumerate(
            zip(peak_df['chr'], peak_df['start'], peak_df['end'])
        )
    }

    gene_peak_map: Dict[str, List[int]] = {}
    for _, row in overlaps_df.iterrows():
        key = row['peak_key']
        if key in peak_to_idx:
            gene_peak_map.setdefault(row['Gene'], []).append(peak_to_idx[key])

    def score_gene(name: str, peaks: List[int], matrix: np.ndarray):
        if not peaks:
            return None
        vals = matrix[:, peaks].sum(axis=1)
        return name, vals

    results = Parallel(n_jobs=-1, prefer='threads')(
        delayed(score_gene)(g, idxs, atac_anndata.X)
        for g, idxs in gene_peak_map.items()
    )
    results = [r for r in results if r]
    genes, mats = zip(*results)
    gene_activity = np.vstack(mats).T  # shape: cells x genes
    print(f"Computed activity for {len(genes)} genes across {gene_activity.shape[0]} cells.")

    if save_in_anndata:
        adata_out = ad.AnnData(X=gene_activity)
        adata_out.var_names = genes
        adata_out.obs_names = atac_anndata.obs_names
        if 'cell_type' in atac_anndata.obs:
            adata_out.obs['cell_type'] = atac_anndata.obs['cell_type']
        return adata_out, gene_activity

    return gene_activity


def filter_based_on_shared_genes(rna_anndata: ad.AnnData, gene_act_anndata: ad.AnnData) -> ad.AnnData:
    if "gene_id" in rna_anndata.var.columns:
        rna_gene_ids = rna_anndata.var["gene_id"].str.upper().str.replace(r'\..*', '', regex=True)
    elif "gene_name" in rna_anndata.var.columns:
        rna_gene_ids = rna_anndata.var["gene_name"].str.upper().str.replace(r'\..*', '', regex=True)
    else:
        raise KeyError("Neither 'gene_id' nor 'gene_name' is in the RNA var.")
    
    activity_genes = gene_act_anndata.var_names.str.upper()
    shared_genes = np.intersect1d(rna_gene_ids.values, activity_genes.values)
    rna_filtered = rna_anndata[:, rna_gene_ids.isin(shared_genes)].copy()
    rna_filtered.obs["CellType"] = rna_filtered.obs["cell_type"]
    print("Filtered RNA shape:", rna_filtered.shape)
    return rna_filtered


def setup_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_path", required=True)
    parser.add_argument("--atac_path", required=True)
    parser.add_argument("--gtf_path", required=True)
    parser.add_argument("--save_dir", default=".")
    return parser


def main() -> None:
    parser = setup_args_parser()
    args = parser.parse_args()

    rna_ad = ad.read_h5ad(args.rna_path)
    atac_ad = ad.read_h5ad(args.atac_path)
    gtf = pr.read_gtf(args.gtf_path)
    gtf_df = gtf.df
    print(f"Head of the GTF file: {gtf_df.head()}")
    
    # Compute gene activity matrix from ATAC data.
    gene_activity_adata, _ = compute_gene_activity(atac_anndata=atac_ad,
                                                     rna_anndata=rna_ad,
                                                     gtf_df=gtf_df,
                                                     save_in_anndata=True)
    print(gene_activity_adata)
    rna_filtered = filter_based_on_shared_genes(rna_anndata=rna_ad, gene_act_anndata=gene_activity_adata)
    print(rna_filtered)
    
    gene_activity_adata.write(os.path.join(args.save_dir, "Gene_Activity.h5ad"))
    rna_filtered.write(os.path.join(args.save_dir, "RNA_filtered.h5ad"))
    
    print("Finished!")


if __name__ == "__main__":
    main()
