from configs import plots
import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sCIN.utils import extract_file_extension
from configs import model_palette, model_order 
import seaborn as sns
import colorcet as cc
from typing import *
import itertools
import os
import re


def _make_palette(models: List[str], palette_name: str = "Paired", fixed_palette: Dict[str, str] = None) -> Dict[str, str]:
    if fixed_palette:
        missing = [m for m in models if m not in fixed_palette]
        print(f"missing: {missing}")
        fallback_colors = sns.color_palette(palette_name, len(missing))
        
        # Cycle through fallback colors, ensuring we have enough for the missing models
        color_cycle = itertools.cycle(fallback_colors)
        
        complete_palette = {}
        for model in models:
            if model in fixed_palette:
                complete_palette[model] = fixed_palette[model]
            else:
                complete_palette[model] = next(color_cycle)  # Get the next color from the cycle
        return complete_palette
    else:
        palette = sns.color_palette(palette_name, len(models))
        return {model: palette[i] for i, model in enumerate(models)}


def _group_data(data_frame:pd.DataFrame, based_on:Union[str, list], 
               select_col:Union[str, list]=None): 

    if select_col is not None:
        return data_frame.groupby(based_on)[select_col]
    
    else:
        return data_frame.groupby(based_on)
    

def _compute_stats_on_grouped_data(grouped_data:pd.DataFrame, stats:Union[str, list]) -> pd.DataFrame:

    return grouped_data.agg(stats).reset_index()


def _draw_err_bars(ax:plt.Axes, grouped_data:pd.DataFrame, category:str, x:str, y:str, colors=Dict[str,str], 
                   err_range:str="std", bar_format:str="-", capsize:int=5, capthick:int=1, linewidth:int=2) -> plt.Axes:

    for item in grouped_data[category].unique():
        item_df = grouped_data[grouped_data[category] == item]
        ax.errorbar(x=item_df[x], y=item_df[y], yerr=item_df[err_range], fmt=bar_format,
                    capsize=capsize, capthick=capthick, linewidth=linewidth, color=colors[item])
        
    return ax


def _handle_legend(
    grouped_data: pd.DataFrame,
    based_on: str,
    colors: Dict[str, str],
    legend_names: Dict[str, str],
    order: List[str],  # Explicit order for alignment
    linewidth: int = 4
) -> list:
    # Create legend entries in the specified order
    legend_handles = [
        Line2D(
            [0], [0],
            color=colors[model],
            linewidth=linewidth,
            label=legend_names.get(model, model)  # Use display name
        )
        for model in order
        if model in grouped_data[based_on].unique()  # Ensure model exists in data
    ]
    return legend_handles


def _draw_legend(ax:plt.Axes, location:str, position:Tuple[float, float], title:str="", num_cols:int=1, 
                font_size:int=18, is_framed:bool=False, columnspacing:float=0.5, handletextpad=0.3, **handle_kwargs) -> plt.Axes:
    
    legend_handles = _handle_legend(**handle_kwargs)
    ax.legend(
        handles=legend_handles, title=title, fontsize=font_size, loc=location, 
        bbox_to_anchor=position, ncols=num_cols, frameon=is_framed, columnspacing=columnspacing, handletextpad=handletextpad
    )

    return ax


def plot_recall_at_k(data_frame: pd.DataFrame, configs: Dict[str, Any], 
                     save_dir: str = None, ax: plt.Axes = None, 
                     fixed_palette: Dict[str, str] = model_palette,
                     plot_inv: bool = False):

    data_frame["Models"] = data_frame["Models"].replace(["AE", "MOFA"], ["Autoencoder", "MOFA+"])
    data_frame["Models"] = data_frame["Models"].replace(["paired", "sCIN_1", "sCIN_5", "sCIN_10", "sCIN_20", "sCIN_50", "sCIN_Random"],                  
                                                        ["Paired", "1%", "5%", "10%", "20%", "50%", "Random"])

    configs = configs["recall_at_k"]
    if ax is None:
        fig, ax = plt.subplots(figsize=(configs["fig_width"], configs["fig_height"]))
    else:
        fig = ax.get_figure()

    if "Recall_at_k_a2r" in data_frame.columns:
        grouped_data = _group_data(data_frame, based_on=["Models", "k"], select_col="Recall_at_k_a2r")
    
    elif plot_inv and "Recall_at_k_r2a" in data_frame.columns:
        grouped_data = _group_data(data_frame, based_on=["Models", "k"], select_col="Recall_at_k_r2a")
    
    else:
        grouped_data = _group_data(data_frame, based_on=["Models", "k"], select_col="Recall_at_k")

    grouped_data_stats = _compute_stats_on_grouped_data(grouped_data, stats=["mean", "std"])

    max_k = grouped_data_stats["k"].max()
    models_order = (
        grouped_data_stats[grouped_data_stats["k"] == max_k]
        .sort_values("mean", ascending=False)["Models"]
        .tolist()
    )
    colors = _make_palette(models=models_order, fixed_palette=fixed_palette)

    ax = _draw_err_bars(ax, grouped_data_stats, category="Models", x="k", y="mean", colors=colors,
                        err_range="std", bar_format=configs["err_bar_format"], 
                        capsize=configs["err_bar_capsize"], capthick=configs["err_bar_capthick"], 
                        linewidth=configs["err_bar_linewidth"])
    
    ax = _draw_legend(ax=ax,
                      location=configs["legend_location"],
                      position=configs["legend_position"],
                      title=configs["legend_title"],
                      num_cols=configs["legend_num_cols"],
                      font_size=configs["legend_fontsize"],
                      is_framed=configs["legend_frame"],
                      columnspacing = configs["legend_columnspacing"],
                      handletextpad = configs["legend_handletextpad"],
                      grouped_data=grouped_data_stats,
                      based_on="Models",
                      colors=colors,
                      legend_names=configs["legend_names"],
                      order=models_order,
                      linewidth=configs["legend_linewidth"])
    
    ax.set_xlabel("k", fontsize=configs["x_axis_fontsize"])
    ax.set_ylabel("Recall@k", fontsize=configs["y_axis_fontsize"])
    xticks_positions = configs["xticks_positions"]
    ax.set_xticks(xticks_positions) 
    ax.set_xticklabels(labels=xticks_positions, fontsize=configs["xticks_fontsize"])
    ax.tick_params(axis="y", labelsize=configs["yticks_fontsize"])
    ax.set_ylim(configs["y_axis_range"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    # out = os.path.join(save_dir, f"RatK_by_models.{configs['file_type']}")
    out = os.path.join(save_dir, f"RatK_by_models")

    plt.savefig(out)
    plt.close(fig)

    return ax


def plot_cell_type_at_k(data_frame: pd.DataFrame, configs: Dict[str, Any], 
                        save_dir: str = None, ax: plt.Axes = None, 
                        fixed_palette: Dict[str, str] = model_palette,
                        plot_inv: bool = False):

    data_frame["Models"] = data_frame["Models"].replace(["AE", "MOFA"], ["Autoencoder", "MOFA+"])
    data_frame["Models"] = data_frame["Models"].replace(["paired", "sCIN_1", "sCIN_5", "sCIN_10", "sCIN_20", "sCIN_50", "sCIN_Random"],                  
                                                        ["Paired", "1%", "5%", "10%", "20%", "50%", "Random"])

    configs = configs["cell_type_at_k"]
    if ax is None:
        fig, ax = plt.subplots(figsize=(configs["fig_width"], configs["fig_height"]))
    else:
        fig = ax.get_figure()

    if "cell_type_at_k_a2r" in data_frame.columns:
        grouped_data = _group_data(data_frame, based_on=["Models", "k"], select_col="cell_type_at_k_a2r")
    
    elif plot_inv and "cell_type_at_k_r2a" in data_frame.columns:
        grouped_data = _group_data(data_frame, based_on=["Models", "k"], select_col="cell_type_at_k_r2a")
    

    grouped_data_stats = _compute_stats_on_grouped_data(grouped_data, stats=["mean", "std"])

    max_k = grouped_data_stats["k"].max()
    models_order = (
        grouped_data_stats[grouped_data_stats["k"] == max_k]
        .sort_values("mean", ascending=False)["Models"]
        .tolist()
    )
    colors = _make_palette(models=models_order, fixed_palette=fixed_palette)

    ax = _draw_err_bars(ax, grouped_data_stats, category="Models", x="k", y="mean", colors=colors,
                        err_range="std", bar_format=configs["err_bar_format"], 
                        capsize=configs["err_bar_capsize"], capthick=configs["err_bar_capthick"], 
                        linewidth=configs["err_bar_linewidth"])
    
    ax = _draw_legend(ax=ax,
                      location=configs["legend_location"],
                      position=configs["legend_position"],
                      title=configs["legend_title"],
                      num_cols=configs["legend_num_cols"],
                      font_size=configs["legend_fontsize"],
                      is_framed=configs["legend_frame"],
                      columnspacing=configs["legend_columnspacing"],
                      grouped_data=grouped_data_stats,
                      based_on="Models",
                      colors=colors,
                      legend_names=configs["legend_names"],
                      order=models_order,
                      linewidth=configs["legend_linewidth"])
    
    ax.set_xlabel("k", fontsize=configs["x_axis_fontsize"])
    ax.set_ylabel("Cell type at k", fontsize=configs["y_axis_fontsize"])
    xticks_positions = configs["xticks_positions"]
    ax.set_xticks(xticks_positions) 
    ax.set_xticklabels(labels=xticks_positions, fontsize=configs["xticks_fontsize"])
    ax.tick_params(axis="y", labelsize=configs["yticks_fontsize"])
    ax.set_ylim(configs["y_axis_range"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    # out = os.path.join(save_dir, f"RatK_by_models.{configs['file_type']}")
    out = os.path.join(save_dir, f"CTatK_by_models")

    plt.savefig(out)
    plt.close(fig)

    return ax


def plot_cell_type_at_k_v1(data_frame: pd.DataFrame, configs: Dict[str, Any], 
                           save_dir: str = None, ax: plt.Axes = None, 
                           fixed_palette: Dict[str, str] = model_palette,
                           plot_inv: bool = False):
    # Rename models for readability
    data_frame = data_frame.copy()
    data_frame["Models"] = data_frame["Models"].replace(
        ["AE", "MOFA", "paired", "sCIN_1", "sCIN_5", "sCIN_10", "sCIN_20", "sCIN_50", "sCIN_Random"],
        ["Autoencoder", "MOFA+", "Paired", "1%", "5%", "10%", "20%", "50%", "Random"],
    )

    configs = configs["cell_type_at_k"]
    if ax is None:
        fig, ax = plt.subplots(figsize=(configs["fig_width"], configs["fig_height"]))
    else:
        fig = ax.get_figure()

    # Choose the appropriate metric
    metric_col = "cell_type_at_k_a2r" if "cell_type_at_k_a2r" in data_frame.columns else "cell_type_at_k_r2a"

    # Group and compute only the mean
    grouped = _group_data(data_frame, based_on=["Models", "k"], select_col=metric_col)
    stats = _compute_stats_on_grouped_data(grouped, stats=["mean"])

    # Determine model order and palette
    max_k = stats["k"].max()
    models_order = (
        stats[stats["k"] == max_k]
        .sort_values("mean", ascending=False)["Models"].tolist()
    )
    colors = _make_palette(models=models_order, fixed_palette=fixed_palette)

    # Plot a line with points for each model, no error bars
    for model in models_order:
        df_model = stats[stats["Models"] == model]
        ax.plot(
            df_model["k"], df_model["mean"],
            marker=configs.get("marker_style", "o"),
            linestyle=configs.get("line_style", "-"),
            linewidth=configs.get("line_width", 2),
            markersize=configs.get("marker_size", 6),
            color=colors[model],
            label=model
        )

    # Draw legend
    ax = _draw_legend(
        ax=ax,
        location=configs["legend_location"],
        position=configs["legend_position"],
        title=configs["legend_title"],
        num_cols=configs["legend_num_cols"],
        font_size=configs["legend_fontsize"],
        is_framed=configs["legend_frame"],
        grouped_data=stats,
        based_on="Models",
        colors=colors,
        legend_names=configs["legend_names"],
        order=models_order,
        linewidth=configs["legend_linewidth"],
    )

    # Axes labels and styling
    ax.set_xlabel("k", fontsize=configs["x_axis_fontsize"])
    ax.set_ylabel("Cell type at k", fontsize=configs["y_axis_fontsize"])
    xt = configs["xticks_positions"]
    ax.set_xticks(xt)
    ax.set_xticklabels([str(x) for x in xt], fontsize=configs["xticks_fontsize"])
    ax.tick_params(axis="y", labelsize=configs["yticks_fontsize"])
    ax.set_ylim(configs["y_axis_range"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(save_dir, "CTatK_by_models_V1")
    plt.savefig(out)
    plt.close(fig)

    return ax


def _plot_boxplot(data_frame: pd.DataFrame, configs: Dict[str, Any], save_dir: str, ax: plt.Axes,
                  y_col: str, order_ascending: bool = False, pre_process: callable = None,
                  fixed_palette: Dict[str, str] = None) -> plt.Axes:

    data_frame["Models"] = data_frame["Models"].replace(["AE", "MOFA"], ["Autoencoder", "MOFA+"])
    data_frame["Models"] = data_frame["Models"].replace(["paired", "sCIN_1", "sCIN_5", "sCIN_10", "sCIN_20", "sCIN_50", "sCIN_Random"],                  
                                                        ["Paired", "1%", "5%", "10%", "20%", "50%", "Random"])
    if pre_process is not None:
        data_frame = pre_process(data_frame)
    
    # Compute model order based on metric
    
    # order = (
    #     data_frame.groupby("Models")[y_col]
    #     .median()
    #     .sort_values(ascending=order_ascending)
    #     .index
    #     .tolist()
    # )
    # if "AE" in order:
    #     idx = order.index("AE")
    #     order[idx] = "Auto Encoder"

    order = [m for m in model_order if m in data_frame["Models"].unique()]

    print(order)
    
    colors = _make_palette(models=order, fixed_palette=fixed_palette)
    
    ax = sns.boxplot(x="Models", y=y_col, data=data_frame, palette=colors, order=order)
    ax.set_xlabel(configs["x_axis_label"])
    ax.set_ylabel(configs["y_axis_label"], fontsize=configs["y_axis_label_fontsize"])

    # legend_handles = _handle_legend(
    #     grouped_data=data_frame,
    #     based_on="Models",
    #     colors=colors,
    #     legend_names=configs["legend_names"],
    #     order=order,  # Pass computed order
    #     linewidth=configs.get("legend_linewidth", 4)
    # )

    # ax.legend(
    #     handles=legend_handles,
    #     title=configs.get("legend_title", ""),
    #     loc=configs.get("legend_location", "best"),
    #     fontsize=configs.get("legend_fontsize", 12)
    # )
    
    if "xticks_positions" in configs:
        ax.set_xticks(configs["xticks_positions"])
        ax.set_xticklabels(order,
                           fontsize=configs["xticks_fontsize"],
                           rotation=configs["xticks_rotation"])
    else:
        ax.set_xticklabels([], fontsize=configs["xticks_fontsize"])
    
    ax.tick_params(axis="y", labelsize=configs["yticks_fontsize"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    
    # out = os.path.join(save_dir, f"{y_col}_by_models.{configs['file_type']}")
    out = os.path.join(save_dir, f"{y_col}_by_models")

    plt.savefig(out)
    
    return ax


def plot_asw(data_frame: pd.DataFrame, configs:Dict[str, Any], 
             save_dir: str = None, ax: plt.Axes = None, fixed_palette: Dict[str, str] = model_palette):
    """
   
    """
    configs = configs["ASW"]
    if ax is None:
        fig, ax = plt.subplots(figsize=(configs["fig_width"], configs["fig_height"]))
    
    else:
        fig = ax.get_figure()
    
    ax = _plot_boxplot(data_frame, configs, save_dir, ax,
                         y_col="cell_type_ASW", order_ascending=False,
                         fixed_palette=fixed_palette)
    
    return ax
    

def plot_cell_type_accuracy(data_frame: pd.DataFrame, 
                            configs:Dict[str, Any], 
                            save_dir:str=None, ax:plt.Axes=None, 
                            fixed_palette: Dict[str, str] = model_palette,
                            plot_inv: bool = False):

    configs = configs["cell_type_accuracy"]
    if ax is None:
        fig, ax = plt.subplots(figsize=(configs["fig_width"], configs["fig_height"]))
    
    if "cell_type_acc_a2r" in data_frame.columns:
        return _plot_boxplot(data_frame, 
                             configs, 
                             save_dir, 
                             ax, 
                             y_col="cell_type_acc_a2r", 
                             order_ascending=False, 
                             fixed_palette=fixed_palette)
    elif plot_inv and "cell_type_acc_r2a" in data_frame.columns:
        return _plot_boxplot(data_frame, 
                             configs, 
                             save_dir, 
                             ax, 
                             y_col="cell_type_acc_r2a", 
                             order_ascending=False, 
                             fixed_palette=fixed_palette)
    else:    
        return _plot_boxplot(data_frame, 
                             configs, 
                             save_dir, 
                             ax, 
                             y_col="cell_type_acc", 
                             order_ascending=False, 
                             fixed_palette=fixed_palette)


def plot_cell_type_accuracy_joint(data_frame: pd.DataFrame, configs: Dict[str, Any],
                                  save_dir: str = None, ax: plt.Axes = None,
                                  fixed_palette: Dict[str, str] = model_palette):
    configs = configs["cell_type_accuracy_joint"]
    if ax is None:
        fig, ax = plt.subplots(figsize=(configs["fig_width"], configs["fig_height"]))
    
    return _plot_boxplot(data_frame, configs, save_dir, ax,
                         y_col="cell_type_acc_joint", order_ascending=False,
                         fixed_palette=fixed_palette)


def _normalize_median_rank(data_frame:pd.DataFrame):

    data_frame["norm_med_rank"] = (data_frame["MedR"] - data_frame["MedR"].min()) / \
        (data_frame["MedR"].max() - data_frame["MedR"].min())
    
    return data_frame
    

def _normalize_median_rank_a2r(data_frame:pd.DataFrame):

    data_frame["norm_med_rank"] = (data_frame["MedR_a2r"] - data_frame["MedR_a2r"].min()) / \
        (data_frame["MedR_a2r"].max() - data_frame["MedR_a2r"].min())
    
    return data_frame


def _normalize_median_rank_r2a(data_frame:pd.DataFrame):

    data_frame["norm_med_rank"] = (data_frame["MedR_r2a"] - data_frame["MedR_r2a"].min()) / \
        (data_frame["MedR_r2a"].max() - data_frame["MedR_r2a"].min())
    
    return data_frame


def plot_median_rank(data_frame: pd.DataFrame, 
                     configs:Dict[str, Any], 
                     save_dir:str=None, 
                     ax:plt.Axes=None, 
                     fixed_palette: Dict[str, str] = model_palette,
                     plot_inv: bool = False):
    
    if "MedR_a2r" in data_frame.columns:
        data_frame = _normalize_median_rank_a2r(data_frame)
    
    elif plot_inv and "MedR_r2a" in data_frame.columns:
        data_frame = _normalize_median_rank_r2a(data_frame)
    
    else:
        data_frame = _normalize_median_rank(data_frame)

    configs = configs["median_rank"]
    if ax is None:
        fig, ax = plt.subplots(figsize=(configs["fig_width"], configs["fig_height"]))
    
    return _plot_boxplot(data_frame, configs, save_dir, ax,
                         y_col="norm_med_rank", order_ascending=False,
                         fixed_palette=fixed_palette)


def plot_GC_joint(data_frame: pd.DataFrame, 
                  configs:Dict[str, Any], 
                  save_dir:str=None, 
                  ax:plt.Axes=None, 
                  fixed_palette: Dict[str, str] = model_palette):
    
    configs = configs["GC_joint"]
    if ax is None:
        fig, ax = plt.subplots(figsize=(configs["fig_width"], configs["fig_height"]))
    
    else:
        fig = ax.get_figure()
    
    ax = _plot_boxplot(data_frame, configs, save_dir, ax,
                       y_col="GC_joint", order_ascending=False,
                       fixed_palette=fixed_palette)
    
    return ax


def compute_tsne_original(mod1_anndata:ad.AnnData, mod2_anndata:ad.AnnData, 
                          num_components:int, test_size:float=0.3, 
                          init:str="random", learning_rate:Union[str, float]="auto") -> Tuple[np.ndarray, np.ndarray]:

    mod1_counts = mod1_anndata.layers["norm_raw_counts"]
    mod2_counts = mod2_anndata.layers["norm_raw_counts"]
    labels = mod1_anndata.obs["cell_type_encoded"]
    _, mod1_test, _, mod2_test, _, labels_test = train_test_split(mod1_counts,
                                                                  mod2_counts,
                                                                  labels, 
                                                                  test_size=test_size,
                                                                  random_state=0)
    all_original = np.column_stack((mod1_test, mod2_test))
    tsne_embs = TSNE(n_components=num_components, learning_rate=learning_rate, 
                     init=init).fit_transform(all_original)
    
    return tsne_embs, labels_test


def _format_cell_types(label:str) -> str:

    label_no_underscore = label.replace("_", "")
    
    lower_label = label_no_underscore.lower()
    if lower_label.startswith("ahigh"):
        label_no_underscore = label_no_underscore[1:]

    elif lower_label.startswith("alow"):
        label_no_underscore = label_no_underscore[1:]
    
    formatted = label_no_underscore[0].upper() + label_no_underscore[1:]

    return formatted


def _make_cell_type_map(cell_types_encoded:pd.Categorical):

    mapping = {i:_format_cell_types(cat) \
               for i, cat in enumerate(cell_types_encoded).cat.categories}
    
    return mapping


def plot_tsne_original(tsne_original:np.array, configs:Dict[str, Any], 
                       seed:int=None, original_labels:np.ndarray=None, 
                       save_dir:str=None, ax:plt.Axes=None, **kwargs):
    
    configs = configs["tSNE_original"]
    test_size = kwargs.get("test_size", 0.3)
    mapping = _make_cell_type_map(original_labels)
    map_func = np.vectorize(lambda x: mapping[x])
    _, labels_test = train_test_split(original_labels, test_size=test_size, random_state=seed)
    labels_mapped = map_func(labels_test)

    colors = cc.glasbey[:len(np.unique(labels_mapped))]
    if ax is None:
        fig, ax = plt.subplots(figsize=(configs["fig_width"], configs["fig_height"]))

    else:
        fig = ax.get_figure()

    for i, cell_type in enumerate(labels_mapped):
        mask = labels_mapped == cell_type
        ax.scatter(tsne_original[mask, 0], tsne_original[mask, 1], 
                    s=configs["point_size_s"], label=f" {cell_type}", color=colors[i])
        
    ax.tick_params(axis="both", which="major", labelsize=configs["tick_fonts"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.subplots_adjust(top=configs["dist_from_top"], 
                       bottom=configs["dist_from_bottom"], hspace=0)
    plt.legend(
        title=configs["title"],
        title_fontsize=configs["title_fontsize"],
        bbox_to_anchor=configs["legend_position"],
        loc=configs["legend_location"],          
        fontsize=configs["legend_fontsize"], 
        ncol=configs["legend_num_cols"],                     
        frameon=configs["is_framed"],
        handleheight=configs["handleheight"],           
        markerscale=configs["marker_scale"],
        columnspacing=configs["column_spacing"],
        labelspacing=configs["label_spacing"]               
    )

    out = os.path.join(save_dir, f"tsne_original.{configs['file_type']}")
    plt.savefig(out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    
    return ax


def compute_tsne_embs(mod1_embs:np.ndarray, mod2_embs:np.ndarray, num_components:int, 
                      init:str="random", learning_rate:Union[str, float]="auto") -> np.array:

    joint_embs = np.column_stack((mod1_embs, mod2_embs))

    return TSNE(n_components=num_components, learning_rate=learning_rate, 
                init=init).fit_transform(joint_embs)


def plot_tsne_embs(tsne_reps:np.ndarray, labels:np.ndarray, configs:Dict[str, Any], 
                   save_dir:str=None, ax:plt.Axes=None) -> plt.Axes:
    
    configs = configs["tSNE_embs"]
    colors = cc.glasbey[:len(np.unique(labels))]

    if ax is None:
        fig, ax = plt.subplots(figsize=(configs["fig_width"], configs["fig_height"]))
    
    else:
        fig = ax.get_figure()

    for i, cell_type in enumerate(labels):
        mask = labels == cell_type
        ax.scatter(tsne_reps[mask, 0], tsne_reps[mask, 1], 
                    s=0.5, label=f" {cell_type}", color=colors[i])
        
    ax.tick_params(axis="both", which="major", labelsize=configs["tick_fonts"])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.subplots_adjust(top=configs["dist_from_top"], 
                       bottom=configs["dist_from_bottom"], hspace=0)

    out = os.path.join(save_dir, f"tsne_original.{configs['file_type']}")
    plt.savefig(out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    
    return ax


def _extract_seed(file_path:str) -> int:
    """
    Extract the random seed value from the file name. 
    """
    file_name = os.path.basename(file_path)
    match = re.findall(r"\d+", file_name)
    if match:
        num = int(match[-1])
        if num in range(0, 100, 10):
            seed = num
        elif num in range(1, 11):
            seed = num * 10

        return seed


def _setup_figure(all_configs: Dict[str, Any], plot_tsne: bool) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Sets up the figure and subplots based on configuration and whether TSNE plots are needed.
    Returns the figure and a dictionary mapping axes names to Axes objects.
    """
    fig = plt.figure(all_configs["fig_width"], all_configs["fig_height"])
    h_space = all_configs.get("horizontal_space", 0.3)
    w_space = all_configs.get("vertical_space", 0.3)
    
    if plot_tsne:
        gs = gridspec.GridSpec(nrows=2, ncols=4, figure=fig, hspace=h_space, wspace=w_space)
    else:
        gs = gridspec.GridSpec(nrows=1, ncols=4, figure=fig, hspace=h_space, wspace=w_space)
    
    axes = {}
    # Top row plots
    axes["top1"] = fig.add_subplot(gs[0, 0])
    axes["top2"] = fig.add_subplot(gs[0, 1])
    axes["top3"] = fig.add_subplot(gs[0, 2])
    axes["top4"] = fig.add_subplot(gs[0, 3])
    
    if plot_tsne:
        # Bottom row TSNE plots
        axes["bot1"] = fig.add_subplot(gs[1, 0:2])
        axes["bot2"] = fig.add_subplot(gs[1, 2:4])
    
    return fig, axes


def _process_tsne_and_plot(
    compute_tsne: bool,
    plot_tsne: bool,
    configs: Dict[str, Any],
    all_configs: Dict[str, Any],
    axes: Dict[str, Any],
    tsne_reps_original_file: Optional[str] = None,
    tsne_reps_embs_file: Optional[str] = None,
    labels_original_file: Optional[str] = None,
    mod1_anndata_file: Optional[str] = None,
    mod2_anndata_file: Optional[str] = None,
    mod1_embs_file: Optional[str] = None,
    mod2_embs_file: Optional[str] = None,
    labels_embs_file: Optional[str] = None,
    num_components: Optional[int] = None,
) -> None:
    """
    Processes TSNE data. If compute_tsne is True, reads the AnnData and embeddings files,
    computes TSNE representations, and plots them. Otherwise, it reads TSNE representations from file.
    """
    if compute_tsne:
        # Read AnnData and get original labels
        mod1_anndata = ad.read_h5ad(mod1_anndata_file)
        mod2_anndata = ad.read_h5ad(mod2_anndata_file)
        labels_original = mod1_anndata.obs["cell_type_encoded"].values
        
        # Save original labels to CSV (ensure you specify an output file name)
        labels_original_df = pd.DataFrame(labels_original, columns=["cell_type_encoded"])
        # Here, you might want to use a provided file name rather than overwriting labels_original_df
        labels_output_file = labels_original_file if labels_original_file else "labels_original.csv"
        labels_original_df.to_csv(labels_output_file, index=False)

        # Load embeddings files based on file extension
        embs_ext = extract_file_extension(mod1_embs_file)
        if embs_ext == ".npy":
            mod1_embs = np.load(mod1_embs_file)
            mod2_embs = np.load(mod2_embs_file)
            labels_embs = np.load(labels_embs_file)
        elif embs_ext == ".csv":
            mod1_embs_df = pd.read_csv(mod1_embs_file)
            mod2_embs_df = pd.read_csv(mod2_embs_file)
            labels_embs_df = pd.read_csv(labels_embs_file)
            mod1_embs = mod1_embs_df.values
            mod2_embs = mod2_embs_df.values
            labels_embs = labels_embs_df.values
        else:
            raise ValueError("Unsupported embedding file extension.")
        
        tsne_original, labels_original_test = compute_tsne_original(mod1_anndata, mod2_anndata, num_components)
        tsne_embs = compute_tsne_embs(mod1_embs, mod2_embs, num_components)

        # seed = _extract_seed(mod1_embs_file)
        plot_tsne_original(tsne_original, configs=all_configs, original_labels=labels_original_test, ax=axes["bot1"])
        plot_tsne_embs(tsne_embs, labels_embs, configs=all_configs, ax=axes["bot2"])
    else:
        # When not computing TSNE, read the TSNE representations from files
        labels_original_df = pd.read_csv(labels_original_file)
        labels_original = labels_original_df.values

        tsne_ext = extract_file_extension(tsne_reps_original_file)
        if tsne_ext == ".npy":
            tsne_original = np.load(tsne_reps_original_file)
            tsne_embs = np.load(tsne_reps_embs_file)
            labels_embs = np.load(labels_embs_file)
        elif tsne_ext == ".csv":
            tsne_original = pd.read_csv(tsne_reps_original_file).values
            tsne_embs = pd.read_csv(tsne_reps_embs_file).values
            labels_embs_df = pd.read_csv(labels_embs_file)
            labels_embs = labels_embs_df.values
        else:
            raise ValueError("Unsupported TSNE file extension.")
        
        # seed = _extract_seed(tsne_reps_original_file)
        plot_tsne_original(tsne_original, 0, labels_original, configs=all_configs, ax=axes["bot1"])
        plot_tsne_embs(tsne_embs, labels_embs, configs=all_configs, ax=axes["bot2"])


def plot_all(
    data_frame: pd.DataFrame,
    save_dir: str,
    configs: Dict[str, Any],
    plot_tsne: bool = True,
    tsne_reps_original_file: Optional[str] = None,
    tsne_reps_embs_file: Optional[str] = None,
    labels_original_file: Optional[str] = None,
    compute_tsne: bool = False,
    mod1_anndata_file: Optional[str] = None,
    mod2_anndata_file: Optional[str] = None,
    mod1_embs_file: Optional[str] = None,
    mod2_embs_file: Optional[str] = None,
    labels_embs_file: Optional[str] = None,
) -> None:
    """
    Main function to generate all plots.
    Breaks the work into two main parts: common plots and TSNE plots (if needed).
    """
    all_configs = configs["all_plots"]

    fig, axes = _setup_figure(all_configs, plot_tsne)

    plot_recall_at_k(data_frame, configs=configs, save_dir=None, ax=axes["top1"])
    plot_asw(data_frame, configs=configs, save_dir=None, ax=axes["top2"])
    plot_cell_type_accuracy(data_frame, configs=configs, save_dir=None, ax=axes["top3"])
    plot_median_rank(data_frame, configs=configs, save_dir=None, ax=axes["top4"])

    if plot_tsne:
        _process_tsne_and_plot(
            compute_tsne,
            plot_tsne,
            configs,
            all_configs,
            axes,
            tsne_reps_original_file=tsne_reps_original_file,
            tsne_reps_embs_file=tsne_reps_embs_file,
            labels_original_file=labels_original_file,
            mod1_anndata_file=mod1_anndata_file,
            mod2_anndata_file=mod2_anndata_file,
            mod1_embs_file=mod1_embs_file,
            mod2_embs_file=mod2_embs_file,
            labels_embs_file=labels_embs_file,
            num_components=configs["tSNE_embs"]["num_components"],
        )

    plt.tight_layout()
    out_path = os.path.join(save_dir, f"all_plots.{all_configs['file_type']}")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)