import os
import argparse
from typing import Tuple, Union

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import colorcet as cc
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE


def _load_array(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    elif ext == ".csv":
        df = pd.read_csv(path, index_col=0)
        return df.values
    else:
        raise ValueError(f"Unsupported file extension for array: {ext}")


def _load_labels(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
        return df.iloc[:, 0].values
    else:
        raise ValueError(f"Unsupported file extension for labels: {ext}")


def compute_tsne(
    X: np.ndarray,
    num_components: int = 2,
    init: str = "random",
    learning_rate: Union[str, float] = "auto",
    seed: int = 0
) -> np.ndarray:
    """
    Compute TSNE embedding for given data matrix X.
    """
    return TSNE(
        n_components=num_components,
        init=init,
        learning_rate=learning_rate,
        random_state=seed
    ).fit_transform(X)


def _plot_tsne(
    tsne_embs: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    figsize: Tuple[float, float],
    point_size: float,
    show_legend: bool,
    legend_fontsize: float,
    legend_markerscale: float,
    top: float,
    bottom: float,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    unique_labels = np.unique(labels)
    colors = cc.glasbey[: len(unique_labels)]
    fig, ax = plt.subplots(figsize=figsize)
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(
            tsne_embs[mask, 0],
            tsne_embs[mask, 1],
            s=point_size,
            label=str(lbl),
            color=colors[i],
            alpha=0.7,
        )
    if show_legend:
        ax.legend(fontsize=legend_fontsize, markerscale=legend_markerscale)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(top=top, bottom=bottom)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def setup_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="t-SNE pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # Compute original data
    pc = sub.add_parser("compute_raw", help="Compute t-SNE on raw AnnData and save embeddings & labels")
    pc.add_argument("--mod1_file", type=str, required=True)
    pc.add_argument("--mod2_file", type=str, required=True)
    pc.add_argument("--save_embs", type=str, required=True, help="Output .npy for embeddings")
    pc.add_argument("--save_labels", type=str, required=True, help="Output .npy for labels")
    pc.add_argument("--num_components", type=int, default=2)
    pc.add_argument("--init", type=str, default="random")
    pc.add_argument(
        "--learning_rate",
        type=lambda x: float(x) if x.lower() != "auto" else "auto",
        default="auto"
    )
    pc.add_argument("--seed", type=int, default=0)

    # Compute on custom embeddings
    pe = sub.add_parser("compute_embs", help="Compute t-SNE on provided embeddings and save")
    pe.add_argument("--emb1_file", type=str, required=True)
    pe.add_argument("--emb2_file", type=str, required=True)
    pe.add_argument("--label1_file", type=str, required=True)
    pe.add_argument("--label2_file", type=str, required=True)
    pe.add_argument("--save_embs", type=str, required=True)
    pe.add_argument("--save_labels", type=str, required=True)
    pe.add_argument("--num_components", type=int, default=2)
    pe.add_argument("--init", type=str, default="random")
    pe.add_argument(
        "--learning_rate",
        type=lambda x: float(x) if x.lower() != "auto" else "auto",
        default="auto"
    )
    pe.add_argument("--seed", type=int, default=0)

    # Plot subcommand
    pp = sub.add_parser("plot", help="Plot from saved embeddings and labels")
    pp.add_argument("--embs_file", type=str, required=True)
    pp.add_argument("--labels_file", type=str, required=True)
    pp.add_argument("--save_dir", type=str, required=True)
    pp.add_argument("--fig_width", type=float, default=6)
    pp.add_argument("--fig_height", type=float, default=8)
    pp.add_argument("--point_size", type=float, default=5)
    pp.add_argument("--show_legend", action="store_true")
    pp.add_argument("--legend_fontsize", type=float, default=10)
    pp.add_argument("--legend_markerscale", type=float, default=2)
    pp.add_argument("--margin_top", type=float, default=0.95)
    pp.add_argument("--margin_bottom", type=float, default=0.05)

    return parser


def main() -> None:
    args = setup_args().parse_args()

    if args.command == "compute_raw":
        data1 = ad.read_h5ad(args.mod1_file)
        data2 = ad.read_h5ad(args.mod2_file)
        labels = data1.obs["cell_type"].values
        idx = np.arange(data1.n_obs)
        _, test_idx = train_test_split(idx, test_size=0.3, random_state=args.seed)
        X1 = data1[test_idx].layers.get("norm_raw_counts", data1[test_idx].X)
        X2 = data2[test_idx].layers.get("norm_raw_counts", data2[test_idx].X)
        joint = np.hstack((X1, X2))
        tsne_embs = compute_tsne(joint, args.num_components, args.init, args.learning_rate, args.seed)
        np.save(args.save_embs, tsne_embs)
        np.save(args.save_labels, labels[test_idx])
        print(f"Saved raw t-SNE embeddings to {args.save_embs} and labels to {args.save_labels}")

    elif args.command == "compute_embs":
        emb1 = _load_array(args.emb1_file)
        emb2 = _load_array(args.emb2_file)
        lbl1 = _load_labels(args.label1_file)
        lbl2 = _load_labels(args.label2_file)
        joint = np.hstack((emb1, emb2))
        labels = np.concatenate([lbl1, lbl2])
        tsne_embs = compute_tsne(joint, args.num_components, args.init, args.learning_rate, args.seed)
        np.save(args.save_embs, tsne_embs)
        np.save(args.save_labels, labels)
        print(f"Saved embedding t-SNE to {args.save_embs} and labels to {args.save_labels}")

    elif args.command == "plot":
        embs = _load_array(args.embs_file)
        labels = _load_labels(args.labels_file)
        save_path = os.path.join(args.save_dir, "tsne_plot.png")
        _plot_tsne(
            embs, labels, save_path,
            (args.fig_width, args.fig_height),
            args.point_size, args.show_legend,
            args.legend_fontsize, args.legend_markerscale,
            args.margin_top, args.margin_bottom
        )
        print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    main()
