import simba as si 
import numpy as np
import pandas as pd 
import anndata as ad
from sCIN.assess import assess
import os 
import logging 
import argparse 


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_file")
    parser.add_argument("--atac_file")
    parser.add_argument("--save_dir")
    parser.add_argument("--num_reps", type=int, choices=range(1, 11), help="Number of replications(1-10)")
    parser.add_argument("--is_inv_metrics", action="store_true")
    parser.add_argument("--quick_test", action="store_true")
    args = parser.parse_args()

    rna_ad = ad.read_h5ad(args.rna_file)
    atac_ad = ad.read_h5ad(args.atac_file)
    labels = rna_ad.obs["cell_type_encoded"]
    dict_adata = {
        "rna":rna_ad,
        "atac":atac_ad
    }

    adata_CP = dict_adata["atac"]  # TODO: Change the "chrom"->"chr", "chromStart"->"start", "chromEnd"->"end"
    adata_CP.var["chr"] = adata_CP.var["chrom"]
    del adata_CP.var["chrom"]
    adata_CP.var["start"] = adata_CP.var["chromStart"]
    del adata_CP.var["chromStart"]
    adata_CP.var["end"] = adata_CP.var["chromEnd"]
    del adata_CP.var["chromEnd"]

    adata_CG = dict_adata["rna"]

    seeds = [seed for seed in range(0, 100, 10)]
    if args.num_reps is not None:
        seeds = seeds[:args.num_reps]

    res = []
    for i, seed in enumerate(seeds):
        rep = i + 1
        adata_CP_iter = adata_CP.copy()
        adata_CG_iter = adata_CG.copy()

        # Keeping peaks associated with top PCs
        si.pp.pca(adata_CP_iter, n_components=50)
        si.pp.select_pcs_features(adata_CP_iter)

        # Discretize RNA expression
        si.tl.discretize(adata_CG_iter,n_bins=5)

        # Compute gene scores
        adata_CG_atac = si.tl.gene_scores(adata_CP_iter,genome="hg38",use_gene_weigt=True, use_top_pcs=True)

        # Infer edges of the graph
        adata_CrnaCatac = si.tl.infer_edges(adata_CG, adata_CG_atac, n_components=15, k=15)

        # Generate graph
        graph_save_dir = os.path.join(args.save_dir, "graphs")
        os.makedirs(graph_save_dir, exist_ok=True)
        si.tl.gen_graph(list_CP=[adata_CP_iter],
                        list_CG=[adata_CG_iter],
                        list_CC=[adata_CrnaCatac],
                        copy=False,
                        use_highly_variable=True,
                        use_top_pcs=True,
                        dirname=graph_save_dir)
        
        # Train the model
        if args.quick_test:
            dict_config = si.settings.pbg_params.copy()
            dict_config["num_epochs"] = 1

        model_save_dir = os.path.join(args.save_dir, "models")
        os.makedirs(model_save_dir, exist_ok=True)
        si.tl.pbg_train(dirname=graph_save_dir,
                        output=model_save_dir)
        
        # Evaluations
        embs_save_dir = os.path.join(args.save_dir, "embs")
        os.makedirs(embs_save_dir, exist_ok=True)
        dict_adata = si.read_embedding()
        atac_embs = dict_adata["C"].X
        # DEBUG
        print(type(atac_embs))
        atac_embs_df = pd.DataFrame(atac_embs)
        atac_embs_df.to_csv(os.path.join(embs_save_dir, f"atac_embs_{rep}.csv"), index=False)

        rna_embs = dict_adata["C2"].X
        rna_embs_df = pd.DataFrame(rna_embs)
        rna_embs_df.to_csv(os.path.join(embs_save_dir, f"rna_embs_{rep}.csv"), index=False)  

        recall_at_k_a2r, num_pairs_a2r, cell_type_acc_a2r, asw, medr_a2r = assess(atac_embs,
                                                                                  rna_embs,
                                                                                  labels,
                                                                                  seed=seed)
        if args.is_inv_metrics:
            recall_at_k_r2a, num_pairs_r2a, cell_type_acc_r2a, _, medr_r2a = assess(rna_embs,
                                                                                    atac_embs,
                                                                                    labels,
                                                                                    seed=seed)
            
        for k, v_a2r in recall_at_k_a2r.items():
            v_r2a = recall_at_k_r2a.get(k, 0)
            res.append({
                "Models":"scButterfly",
                "Replicates":i+1,
                "k":k,
                "Recall_at_k_a2r":v_a2r,
                "Recall_at_k_r2a":v_r2a if v_r2a is not None else 0.0,
                "num_pairs_a2r":num_pairs_a2r,
                "num_pairs_r2a":num_pairs_r2a if num_pairs_r2a is not None else 0.0,
                "cell_type_acc_a2r":cell_type_acc_a2r,
                "cell_type_acc_r2a":cell_type_acc_r2a if cell_type_acc_r2a is not None else 0.0,
                "cell_type_ASW":asw,
                "MedR_a2r":medr_a2r,
                "MedR_r2a":medr_r2a if medr_r2a is not None else 0.0
            })

    results = pd.DataFrame(res)
    res_save_dir = os.path.join(args.save_dir, "outs")
    os.makedirs(res_save_dir, exist_ok=True)
    results.to_csv(os.path.join(res_save_dir, f"metrics_SIMBA_{args.num_reps}reps.csv"), index=False)
    logging.info(f"SIMBA results saved to {res_save_dir}.")

        
if __name__ == "__main__":
    main()