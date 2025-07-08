"""
Usage:

t-SNAE on model's embs:

python tsne_pipeline.py compute_embs \
  --emb1_file path/to/model1_embs.npy \
  --emb2_file path/to/model2_embs.csv \
  --label1_file path/to/model1_labels.csv \
  --label2_file path/to/model2_labels.npy \
  --save_embs path/to/tsne_model.npy \
  --save_labels path/to/tsne_model_labels.npy \
  --paired (or --unpaired)
  --num_components 2 \
  --init pca \
  --learning_rate auto \
  --seed 0

Plot t-SNE embs:
  
python tsne_pipeline.py plot \
  --embs_file path/to/tsne_model.npy \
  --labels_file path/to/tsne_model_labels.npy \
  --save_dir path/to/plots \
  --show_legend

t-SNE on original data:

python tsne_pipeline.py compute_raw \
  --mod1_file path/to/raw_mod1.h5ad \
  --mod2_file path/to/raw_mod2.h5ad \
  --save_embs results/raw_tsne_embs.npy \
  --save_labels results/raw_tsne_labels.npy \
  --paired (or --unpaired)
  --num_components 2 \
  --init pca \
  --learning_rate auto \
  --seed 0

Plot t-SNE of the original data:
  
python tsne_pipeline.py plot \
  --embs_file results/model_tsne_embs.npy \
  --labels_file results/model_tsne_labels.npy \
  --save_dir results/plots/model \
  --show_legend

"""

import os
import argparse
from typing import Tuple, Union
import matplotlib as mpl
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib as mpl
import colorcet as cc
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


mpl.rcParams["figure.dpi"] = 1200
mpl.rcParams["savefig.dpi"] = 1200

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
        return np.load(path, allow_pickle=True)
    elif ext == ".csv":
        df = pd.read_csv(path)
        return df.iloc[:, 0].values
    else:
        raise ValueError(f"Unsupported file extension for labels: {ext}")


def compute_tsne(
    X: np.ndarray,
    num_components: int,
    init: str,
    learning_rate: Union[str, float],
    seed: int
) -> np.ndarray:
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
        # If a mapping file for numeric codes exists, load it to convert labels
    map_path = os.path.join(os.path.dirname(save_path), 'mapped_cell_types.txt')
    code_to_name = {}
    if os.path.exists(map_path):
        with open(map_path) as mf:
            for line in mf:
                line = line.strip()
                if not line: continue
                # Expect format code:Name or code	Name
                parts = line.replace('	',':').split(':', 1)
                code, name = parts[0], parts[1]
                code_to_name[code] = name
    # Clean and format labels, mapping codes if needed
    formatted = []
    for lbl in labels:
        lbl_str = str(lbl)
        # map numeric code to name if available
        if lbl_str in code_to_name:
            s = code_to_name[lbl_str]
        else:
            s = lbl_str
        # replace underscores and adjust 'ahigh'/'alow'
        s = s.replace('_', ' ')
        low = s.lower()
        if low.startswith('ahigh'):
            s = 'a high' + s[5:]
        elif low.startswith('alow'):
            s = 'a low' + s[4:]
        # capitalize first character
        if s:
            s = s[0].upper() + s[1:]
        formatted.append(s)
    clean_labels = np.array(formatted)
    unique_labels = np.unique(clean_labels)
    
    # Deterministic color mapping: use existing mapping file if present, else generate & save
    mapping_file = os.path.join(os.path.dirname(save_path), 'label_color_mapping.txt')
    if os.path.exists(mapping_file):
        label_color_mapping = {}
        with open(mapping_file) as f:
            for line in f:
                parts = line.strip().split('	', 1)  # split on tab only
                if len(parts) != 2:
                    continue
                lbl_key, hexcol = parts
                label_color_mapping[lbl_key] = hexcol
        
        missing = [lbl for lbl in unique_labels if lbl not in label_color_mapping]
        if missing:
            cmap = cc.cm['glasbey_light']
            start = len(label_color_mapping)
            for i, m_lbl in enumerate(missing, start=start):
                col = cmap(i / max(len(unique_labels)-1, 1))
                label_color_mapping[m_lbl] = mpl.colors.to_hex(col)
            
            with open(mapping_file, 'w') as f:
                for lbl_key, hexcol in label_color_mapping.items():
                    f.write(f"{lbl_key}	{hexcol}")

        colors = [mpl.colors.to_rgb(label_color_mapping[lbl]) for lbl in unique_labels]
    
    else:
        cmap = cc.cm['glasbey_light']  # high-contrast categorical map
        colors = [cmap(i / max(len(unique_labels)-1, 1)) for i in range(len(unique_labels))]
        label_color_mapping = {lbl: mpl.colors.to_hex(col) for lbl, col in zip(unique_labels, colors)}
        with open(mapping_file, 'w') as f:
            for lbl_key, hexcol in label_color_mapping.items():
                f.write(f"{lbl_key}	{hexcol}\n")

    fig, ax = plt.subplots(figsize=figsize)
    for i, lbl in enumerate(unique_labels):
        mask = clean_labels == lbl
        ax.scatter(
            tsne_embs[mask, 0],
            tsne_embs[mask, 1],
            s=point_size,
            label=lbl,
            color=colors[i],
            alpha=0.8,
        )
    if show_legend:
        ax.legend(
            fontsize=legend_fontsize,
            markerscale=legend_markerscale,
            frameon=False,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.2),
            ncol=min(len(unique_labels), 5)
        )
    ax.tick_params(axis='both', labelsize=legend_fontsize)
    ax.spines["left"].set_linewidth(3)   
    ax.spines["bottom"].set_linewidth(3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis="x", labelsize=25)
    ax.tick_params(axis="y", labelsize=25)
    fig.subplots_adjust(top=top, bottom=bottom)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def setup_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="t-SNE pipeline with PCA for unpaired alignment")
    sub = parser.add_subparsers(dest='command', required=True)

    # raw data compute
    pc = sub.add_parser('compute_raw', help='Compute t-SNE on raw AnnData')
    pc.add_argument('--mod1_file', required=True)
    pc.add_argument('--mod2_file', required=True)
    pc.add_argument('--save_embs', required=True)
    pc.add_argument('--save_labels', required=True)
    pc.add_argument('--paired', action='store_true', help='Treat data as paired')
    pc.add_argument('--unpaired', action='store_true', help='Treat data as unpaired')
    pc.add_argument('--pca_components', type=int, default=50,
                    help='Number of PCA components for unpaired alignment')
    pc.add_argument('--num_components', type=int, default=2)
    pc.add_argument('--init', default='pca')
    pc.add_argument('--learning_rate', type=lambda x: float(x) if x!='auto' else 'auto', default='auto')
    pc.add_argument('--seed', type=int, default=0)

    # model embeddings compute
    pe = sub.add_parser('compute_embs', help='Compute t-SNE on model embeddings')
    pe.add_argument('--emb1_file', required=True)
    pe.add_argument('--emb2_file', required=True)
    pe.add_argument('--label1_file', required=True)
    pe.add_argument('--label2_file', required=True)
    pe.add_argument('--save_embs', required=True)
    pe.add_argument('--save_labels', required=True)
    pe.add_argument('--paired', action='store_true')
    pe.add_argument('--unpaired', action='store_true')
    pe.add_argument('--num_components', type=int, default=2)
    pe.add_argument('--init', default='random')
    pe.add_argument('--learning_rate', type=lambda x: float(x) if x!='auto' else 'auto', default='auto')
    pe.add_argument('--seed', type=int, default=0)

    # plot
    pp = sub.add_parser('plot', help='Plot from saved embeddings & labels')
    pp.add_argument('--embs_file', required=True)
    pp.add_argument('--labels_file', required=True)
    pp.add_argument('--save_dir', required=True)
    pp.add_argument('--plot_name', type=str, default='tsne_plot.png',
                    help='Filename for the saved plot (e.g. plot_raw.png)')
    pp.add_argument('--fig_width', type=float, default=6)
    pp.add_argument('--fig_height', type=float, default=6)
    pp.add_argument('--matplot_backend', type=str, default='AGG')
    pp.add_argument('--fig_dpi', type=int, default=300)
    pp.add_argument('--savefig_dpi', type=int, default=300)
    pp.add_argument('--point_size', type=float, default=1)
    pp.add_argument('--show_legend', action='store_true')
    pp.add_argument('--legend_fontsize', type=float, default=12)
    pp.add_argument('--legend_markerscale', type=float, default=6)
    pp.add_argument('--margin_top', type=float, default=0.95)
    pp.add_argument('--margin_bottom', type=float, default=0.15)

    return parser


def main() -> None:
    args = setup_args().parse_args()
    # mpl.rcParams['backend'] = args.matplot_backend
    mpl.rcParams['figure.dpi'] = 1200
    mpl.rcParams['savefig.dpi'] = 1200

    if args.command == 'compute_raw':
        data1 = ad.read_h5ad(args.mod1_file)
        data2 = ad.read_h5ad(args.mod2_file)
        X1 = data1.layers.get('norm_raw_counts', data1.X)
        X2 = data2.layers.get('norm_raw_counts', data2.X)
        lbl1 = data1.obs['cell_type_encoded'].values
        lbl2 = data2.obs['cell_type_encoded'].values

        if args.paired:
            _, idx = train_test_split(np.arange(X1.shape[0]), test_size=0.3, random_state=args.seed)
            joint = np.hstack((X1[idx], X2[idx]))
            labels = lbl1[idx]
        elif args.unpaired:
            # reduce via PCA
            pca = PCA(n_components=args.pca_components, random_state=args.seed)
            X1r = pca.fit_transform(X1)
            X2r = pca.fit_transform(X2)
            joint = np.vstack((X1r, X2r))
            labels = np.concatenate((lbl1, lbl2))
        else:
            raise ValueError('Specify --paired or --unpaired')

        embs = compute_tsne(joint, args.num_components, args.init, args.learning_rate, args.seed)
        os.makedirs(os.path.dirname(args.save_embs), exist_ok=True)
        os.makedirs(os.path.dirname(args.save_labels), exist_ok=True)
        np.save(args.save_embs, embs)
        np.save(args.save_labels, labels)
        print(f"Saved raw t-SNE embeddings to {args.save_embs} and labels to {args.save_labels}")

    elif args.command == 'compute_embs':
        E1 = _load_array(args.emb1_file)
        E2 = _load_array(args.emb2_file)
        L1 = _load_labels(args.label1_file)
        L2 = _load_labels(args.label2_file)
        if args.paired:
            joint = np.hstack((E1, E2))
            labels = L1
        elif args.unpaired:
            joint = np.vstack((E1, E2))
            labels = np.concatenate((L1, L2))
        else:
            raise ValueError('Specify --paired or --unpaired')
        embs = compute_tsne(joint, args.num_components, args.init, args.learning_rate, args.seed)
        os.makedirs(os.path.dirname(args.save_embs), exist_ok=True)
        os.makedirs(os.path.dirname(args.save_labels), exist_ok=True)
        np.save(args.save_embs, embs)
        np.save(args.save_labels, labels)
        print(f"Saved model t-SNE embeddings to {args.save_embs} and labels to {args.save_labels}")

    elif args.command == 'plot':
        embs = _load_array(args.embs_file)
        labels = _load_labels(args.labels_file)
        # use user-defined plot name
        save_path = os.path.join(args.save_dir, args.plot_name)
        _plot_tsne(
            embs, labels, save_path,
            (args.fig_width, args.fig_height),
            args.point_size, args.show_legend,
            args.legend_fontsize, args.legend_markerscale,
            args.margin_top, args.margin_bottom
        )
        print(f"Saved plot to {save_path}")

if __name__ == '__main__':
    main()