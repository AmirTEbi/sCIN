from configs import plots
import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import seaborn as sns
import colorcet as cc
from typing import Union, List, Dict, Tuple, Any
import os


def _make_palette(models:List[str], palette_name:str="Paired") -> Dict[str:str]:

    palette = sns.color_palette(palette_name, len(models))
    
    return {model:palette[i] for i, model in enumerate(models)}


def _group_data(data_frame:pd.DataFrame, based_on:Union[str, list], 
               select_col:Union[str, list]=None) -> pd.core.groupby.generic.DataFrameGroupBy:

    if select_col is not None:
        return data_frame.groupby(based_on)[select_col]
    
    else:
        return data_frame.groupby(based_on)
    

def _compute_stats_on_grouped_data(grouped_data:pd.DataFrame, stats:Union[str, list]) -> pd.DataFrame:

    return grouped_data.agg(stats).reset_index()


def _draw_err_bars(ax:plt.Axes, grouped_data:pd.DataFrame, category:str, x:str, y:str, colors=Dict[str:str], 
                  err_range:str="std", bar_format:str="-", capsize:int=5, capthick:int=1, linewidth:int=2) -> plt.Axes:

    for item in grouped_data[category].unique():
        item_df = grouped_data[grouped_data[category] == item]
        ax.errorbar(x=item_df[x], y=item_df[y], yerr=item_df[err_range], fmt=bar_format,
                    capsize=capsize, capthick=capthick, linewidth=linewidth, color=colors[item])
        
    return ax


def _handle_legend(grouped_data:pd.DataFrame, based_on:str, colors=Dict[str:str], 
                   legend_names:Dict[str:str]=None, linewidth:int=4) -> list:

    legend_handles = [
        Line2D(
            [0], [0], color=colors[item], linewidth=linewidth, 
            label=legend_names.get(item, item)  
        ) for item in grouped_data[based_on].unique()
    ]

    return legend_handles


def _draw_legend(ax:plt.Axes, location:str, position:Tuple[float, float], title:str="", num_cols:int=1, 
                font_size:int=18, is_framed:bool=False, **handle_kwargs) -> plt.Axes:
    
    legend_handles = _handle_legend(**handle_kwargs)
    ax.legend(
        handles=legend_handles, title=title, fontsize=font_size, loc=location, 
        bbox_to_anchor=position, ncols=num_cols, frameon=is_framed
    )

    return ax


def plot_recall_at_k(data_frame:pd.DataFrame, configs:Dict[str:Any], 
                     save_dir:str=None, ax:plt.Axes=None):
    
    configs = configs["recall_at_k"]
    if ax is None:
        fig, ax = plt.subplots(configs["fig_size"])

    else:
        fig = ax.get_figure()
    
    models = data_frame["Models"].unique().tolist()
    colors = _make_palette(models)

    grouped_data = _group_data(data_frame, based_on=["Models", "k"], select_col="Recall_at_k")
    grouped_data_stats = _compute_stats_on_grouped_data(grouped_data, stats=["mean", "std"])

    ax = _draw_err_bars(ax, grouped_data_stats, category="Models", x="k", y="mean", colors=colors,
                        err_range="std", bar_format=configs["err_bar_format"], 
                        capsize=configs["err_bar_capsize"], capthick=configs["err_bar_capthick"], 
                        linewidth=configs["err_bar_linewidth"])
    
    ax = _draw_legend(location=configs["legend_location"],
                      position=configs["legend_position"],
                      title=configs["legend_title"],
                      num_cols=configs["legend_num_cols"],
                      font_size=configs["legend_fontsize"],
                      is_framed=configs["legend_frame"],
                      grouped_data=grouped_data_stats,
                      based_on="Models",
                      colors=colors,
                      legend_names=configs["legend_names"],
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
    out = os.path.join(save_dir, f"RatK_by_models.{configs['file_type']}")
    plt.savefig(out)
    plt.close(fig)

    return ax


def _plot_boxplot(data_frame:pd.DataFrame, configs:Dict[str:Any], save_dir:str, ax:plt.Axes,
                  y_col:str, pre_process:callable=None) -> plt.Axes:
    """
    A helper function to draw a boxplot.
    """
    if pre_process is not None:
        data_frame = pre_process(data_frame)
    
    models = data_frame["Models"].unique().tolist()
    colors = _make_palette(models)
    
    ax = sns.boxplot(x="Models", y=y_col, data=data_frame, palette=colors)
    
    ax.set_xlabel(configs["x_axis_label"])
    ax.set_ylabel(configs["y_axis_label"], fontsize=configs["y_axis_label_fontsize"])
    
    if "xticks_positions" in configs:
        ax.set_xticks(configs["xticks_positions"])
        ax.set_xticklabels(configs["xticks_labels"],
                           fontsize=configs["xticks_fontsize"],
                           rotation=configs["xticks_rotation"])
    else:
        ax.set_xticklabels([], fontsize=configs["xticks_fontsize"])
    
    ax.tick_params(axis="y", labelsize=configs["yticks_fontsize"])
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    
    out = os.path.join(save_dir, f"{y_col}_by_models.{configs['file_type']}")
    plt.savefig(out)
    
    return ax


def plot_asw(data_frame: pd.DataFrame, configs:Dict[str:Any], 
             save_dir: str = None, ax: plt.Axes = None):
    """
   
    """
    configs = configs["ASW"]
    if ax is None:
        fig, ax = plt.subplots(figsize=configs["fig_size"])
    
    else:
        fig = ax.get_figure()
    
    ax = _plot_boxplot(data_frame, configs, save_dir, ax, y_col="cell_type_ASW")
    
    return ax
    

def plot_cell_type_accuracy(data_frame: pd.DataFrame, configs:Dict[str:Any], 
                            save_dir:str=None, ax:plt.Axes=None):

    configs = configs["cell_type_accuracy"]
    if ax is None:
        fig, ax = plt.subplots(configs["fig_size"])
    
    return _plot_boxplot(data_frame, configs, save_dir, ax, y_col="cell_type_acc")



def normalize_median_rank(data_frame:pd.DataFrame):

    data_frame["norm_med_rank"] = (data_frame["MedR"] - data_frame['MedR'].min()) / \
        (data_frame["MedR"].max() - data_frame["MedR"].min())
    
    return data_frame
    

def plot_median_rank(data_frame: pd.DataFrame, configs:Dict[str:Any], 
                     save_dir:str=None, ax:plt.Axes=None):

    configs = configs["cell_type_accuracy"]
    if ax is None:
        fig, ax = plt.subplots(configs["fig_size"])

    else:
        fig = ax.get_figure()
    
    return _plot_boxplot(data_frame, configs, save_dir, ax, 
                         y_col="cell_type_acc", pre_process=normalize_median_rank)


def compute_tsne_original(mod1_anndata:ad.AnnData, mod2_anndata:ad.AnnData, num_components:int, 
                          test_size:float=0.3, init:str="random", learning_rate:Union[str, float]="auto") -> np.ndarray:

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


def format_cell_types(label:str) -> str:

    label_no_underscore = label.replace("_", "")
    
    lower_label = label_no_underscore.lower()
    if lower_label.startswith("ahigh"):
        label_no_underscore = label_no_underscore[1:]

    elif lower_label.startswith("alow"):
        label_no_underscore = label_no_underscore[1:]
    
    formatted = label_no_underscore[0].upper() + label_no_underscore[1:]

    return formatted


def make_cell_type_map(cell_types_encoded:pd.Categorical):

    mapping = {i:format_cell_types(cat) \
               for i, cat in enumerate(cell_types_encoded).cat.categories}
    
    return mapping


def compute_plot_tsne_original(mod1_anndata:ad.AnnData, mod2_anndata:ad.AnnData, 
                               configs:dict, save_dir:str=None, ax:plt.Axes=None, **kwargs):
    
    configs = configs["tSNE_original"]
    tsne_reps, labels = compute_tsne_original(mod1_anndata, mod2_anndata, 
                                              num_components=configs["num_components"], 
                                              test_size=configs["test_size"])
    mapping = make_cell_type_map(mod1_anndata.obs["cell_type_encoded"])
    map_func = np.vectorize(lambda x: mapping[x])
    labels_mapped = map_func(labels)

    colors = cc.glasbey[:len(np.unique(labels_mapped))]

    if ax is None:
        fig, ax = plt.subplots(configs["fig_size"])

    else:
        fig = ax.get_figure()

    for i, cell_type in enumerate(labels_mapped):
        mask = labels_mapped == cell_type
        ax.scatter(tsne_reps[mask, 0], tsne_reps[mask, 1], 
                    s=0.5, label=f" {cell_type}", color=colors[i])
        
    ax.tick_params(axis="both", which="major", labelsize=configs["tick_fonts"])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

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


def plot_tsne_original(tsne_original:np.array, seed:int, mod_anndata:ad.AnnData, 
                       configs:Dict[str:Any], save_dir:str=None, ax:plt.Axes=None, **kwargs):
    
    configs = configs["tSNE_original"]
    test_size = kwargs.get("test_size", 0.3)
    mapping = make_cell_type_map(mod_anndata.obs["cell_type_encoded"])
    map_func = np.vectorize(lambda x: mapping[x])
    labels = mod_anndata.obs["cell_type_encoded"]
    _, labels_test = train_test_split(labels, test_size=test_size, random_state=seed)
    labels_mapped = map_func(labels_test)

    colors = cc.glasbey[:len(np.unique(labels_mapped))]
    if ax is None:
        fig, ax = plt.subplots(configs["fig_size"])

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


def compute_plot_tsne_embs(mod1_embs:np.ndarray, mod2_embs:np.ndarray, labels:np.ndarray,
                           configs:Dict[str:Any], save_dir:str=None, ax:plt.Axes=None) -> plt.Axes:
    
    configs = configs["tSNE_embs"]
    colors = cc.glasbey[:len(np.unique(labels))]
    tsne_embs = compute_tsne_embs(mod1_embs, 
                                  mod2_embs, 
                                  num_components=configs["num_components"], 
                                  init=configs["init_method"], 
                                  learning_rate=configs["learning_rate"])

    if ax is None:
        fig, ax = plt.subplots(configs["fig_size"])
    
    else:
        fig = ax.get_figure()

    for i, cell_type in enumerate(labels):
        mask = labels == cell_type
        ax.scatter(tsne_embs[mask, 0], tsne_embs[mask, 1], 
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


def plot_tsne_embs(tsne_reps:np.ndarray, labels:np.ndarray, configs:Dict[str:Any], 
                   save_dir:str=None, ax:plt.Axes=None) -> plt.Axes:
    
    configs = configs["tSNE_embs"]
    colors = cc.glasbey[:len(np.unique(labels))]

    if ax is None:
        fig, ax = plt.subplots(configs["fig_size"])
    
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
        

def plot_all(data_frame: pd.DataFrame,
             mod1_anndata: ad.AnnData,
             mod2_anndata: ad.AnnData,
             mod1_embs: np.ndarray,
             mod2_embs: np.ndarray,
             labels: np.ndarray,
             save_dir: str,
             configs: Dict[str:Any],
             is_tsne: bool = True):
    
    all_configs = configs["all_plots"]
    
    fig = plt.figure(figsize=all_configs["fig_size"])
    if is_tsne:
        gs = gridspec.GridSpec(nrows=2, ncols=4, figure=fig, 
                               hspace=all_configs.get("horizontal_space", 0.3), 
                               wspace=all_configs.get("vertical_space", 0.3))
    else:
        gs = gridspec.GridSpec(nrows=1, ncols=4, figure=fig, 
                               hspace=all_configs.get("horizontal_space", 0.3), 
                               wspace=all_configs.get("vertical_space", 0.3))
    
    ax_top1 = fig.add_subplot(gs[0, 0])
    ax_top2 = fig.add_subplot(gs[0, 1])
    ax_top3 = fig.add_subplot(gs[0, 2])
    ax_top4 = fig.add_subplot(gs[0, 3])
    
    if is_tsne:
        ax_bot1 = fig.add_subplot(gs[1, 0:2])
        ax_bot2 = fig.add_subplot(gs[1, 2:4])
    
    plot_recall_at_k(data_frame, configs=configs, save_dir=None, ax=ax_top1)
    plot_asw(data_frame, configs=configs, save_dir=None, ax=ax_top2)
    plot_cell_type_accuracy(data_frame, configs=configs, save_dir=None, ax=ax_top3)
    plot_median_rank(data_frame, configs=configs, save_dir=None, ax=ax_top4)

    if is_tsne:
        plot_tsne_original(mod1_anndata, mod2_anndata, configs=configs, save_dir=None, ax=ax_bot1)
        plot_tsne_embs(mod1_embs, mod2_embs, labels, configs=configs, save_dir=None, ax=ax_bot2)

    plt.tight_layout()
    out = os.path.join(save_dir, f"all_plots.{all_configs['file_type']}")
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)