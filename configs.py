import seaborn as sns

# ========= Models ========= 

MOFA = {
    "num_replications":10,
    "num_factors":30,
    "num_iterations":1000,
    "num_PCs":100,
    "convergence_mode":"fast"
}

Harmony = {
    "max_iter_harmony":10,
    "PCs":100,
}

AutoEncoder = {
    "num_epochs":150,
    "hidden_dim":256,
    "latent_dim":128,
    "learning_rate":0.01,
}

sCIN = {
    "num_epochs":100,
    "t":0.07,
    "learning_rate":0.001,
    "hidden_dim":256,
    "latent_dim":128,
    "bob":10,
    "patience":10,
    "min_delta":1e-4
}

ConAAE = {
    "PCs":100,
    "num_epochs":100
}

# ========= Plots ========= 


# model_order = ["sCIN", "Con-AAE", "scBridge", "scGLUE", "sciCAN", "MOFA+", "Autoencoder", "Harmony"]
model_order = ["Random", "1%", "5%", "10%", "20%", "50%", "Paired"]


# model_palette = {
#     "Con-AAE": "#e41a1c",
#     "sCIN": "#377eb8",
#     "scBridge": "#4daf4a",
#     "scGLUE":"#984ea3",
#     "MOFA+": "#ff7f00",
#     "Autoencoder":"#a65628",        
#     "Harmony": "#a6cee3",
#     "sciCAN": "#666666"
# }

model_palette = {
    "Random":"#a6cee3",
    "1%":"#1f78b4",
    "5%":"#b2df8a",
    "10%":"#33a02c",
    "20%":"#fb9a99",
    "50%":"#e31a1c",
    "Paired":"#fdbf6f" 
}


plots = {
    "recall_at_k":{

        "fig_width":11.44,
        "fig_height":7.53,
        "err_bar_format":"-",
        "err_bar_capsize":5,
        "err_bar_capthick":3,
        "err_bar_linewidth":4,
        "legend_names":{
            "sCIN":"sCIN",
            "Con-AAE":"Con-AAE",
            "MOFA+":"MOFA+",
            "Harmony":"Harmony",
            "Autoencoder":"Autoencoder",
            "scBridge":"scBridge",
            "scGLUE":"scGLUE",
            "sciCAN":"sciCAN"
        },
        "legend_location":"lower center",
        "legend_position":(0.43, 0.8),
        "legend_title":"",
        "legend_num_cols":4,
        "legend_fontsize":26,
        "legend_frame":False,
        "legend_linewidth":5,
        "legend_columnspacing":0.5,
        "legend_handletextpad":14,
        "x_axis_fontsize":35,
        "y_axis_fontsize":35,
        "x_axis_linewidth":4,
        "x_tick_pointer_width":4,
        "y_tick_pointer_width":4,
        "y_axis_linewidth":4,
        "xticks_positions":[10, 20, 30, 40, 50],
        "xticks_fontsize":30,
        "yticks_fontsize":30,
        "y_axis_range":(0, 0.40)

    },

    "cell_type_at_k":{

        "fig_width":17.76,
        "fig_height":8.639,
        "err_bar_format":"-",
        "err_bar_capsize":5,
        "err_bar_capthick":3,
        "err_bar_linewidth":4,
        "legend_names":{
            "sCIN":"sCIN",
            "Con-AAE":"Con-AAE",
            "MOFA+":"MOFA+",
            "Harmony":"Harmony",
            "Autoencoder":"Autoencoder",
            "scBridge":"scBridge",
            "scGLUE":"scGLUE",
            "sciCAN":"sciCAN"
        },
        "legend_location":"lower center",
        "legend_position":(0.51, 0.85),
        "legend_title":"",
        "legend_num_cols":4,
        "legend_fontsize":26,
        "legend_frame":False,
        "legend_linewidth":5,
        "legend_columnspacing":0.5,
        "legend_handletextpad":10,
        "x_axis_linewidth":4,
        "y_axis_linewidth":4,
        "x_axis_fontsize":35,
        "y_axis_fontsize":35,
        "xticks_positions":[10, 20, 30, 40, 50],
        "xticks_fontsize":30,
        "yticks_fontsize":30,
        "x_tick_pointer_width":4,
        "y_tick_pointer_width":4,
        "y_axis_range":(0, 1.15)

    },

    "ASW":{

        "fig_width":11.52,
        "fig_height":8.23,
        "x_axis_label":"",
        "y_axis_label":"ASW",
        "y_axis_label_fontsize":35,
        "y_label_fontsize":35,
        "xticks_positions":[0, 1, 2, 3, 4, 5, 6],
        "xticks_labels":['Con-AAE', 'sCIN', 'scBridge', 'scGLUE', 'sciCAN', 'MOFA+', 'Autoencoder', 'Harmony'],
        "x_axis_linewidth":4,
        "y_axis_linewidth":4,
        "x_tick_pointer_width":4,
        "y_tick_pointer_width":4,
        "xticks_fontsize":30,
        "xticks_rotation":45,
        "yticks_fontsize":30,
        "y_axis_range":(0.5, 0.65)

    },

    "GC_joint":{

        "fig_width":11.52,
        "fig_height":6.03,
        "x_axis_label":"",
        "y_axis_label":"Graph Connectivity",
        "y_axis_label_fontsize":26,
        "y_label_fontsize":26,
        "xticks_positions":[0, 1, 2, 3, 4, 5, 6],
        "xticks_labels":['Con-AAE', 'sCIN', 'scBridge', 'scGLUE', 'sciCAN', 'MOFA+', 'Autoencoder', 'Harmony'],
        "xticks_fontsize":22,
        "xticks_rotation":45,
        "yticks_fontsize":22,
        "y_axis_range":(0, 0.8)

    },

    "cell_type_accuracy":{

        "fig_width":11.52,
        "fig_height":8.23,
        "x_axis_label":"",
        "y_axis_label":"Cell Type Accuracy",
        "y_axis_label_fontsize":35,
        "y_axis_range":(0, 1),
        "xticks_positions":[0, 1, 2, 3, 4, 5, 6],
        "xticks_labels":["sCIN", "Con-AAE", "Harmony", "MOFA+", "Autoencoder", "scGLUE", "sciCAN"],
        "x_axis_linewidth":4,
        "y_axis_linewidth":4,
        "x_tick_pointer_width":4,
        "y_tick_pointer_width":4,
        "xticks_fontsize":30,
        "xticks_rotation":45,
        "yticks_fontsize":30
        
    },

    "cell_type_accuracy_joint":{

        "fig_width":8,
        "fig_height":6,
        "x_axis_label":"",
        "y_axis_label":"Joint Cell Type Accuracy",
        "y_axis_label_fontsize":20,
        "y_axis_range":(0, 0.8),
        "xticks_positions":[0, 1, 2, 3, 4, 5, 6],
        "xticks_labels":["Con-AAE", "scBridge", "sCIN", "MOFA", "AE", "Harmony"],
        "xticks_fontsize":18,
        "xticks_rotation":45,
        "yticks_fontsize":18
        
    },

    "median_rank":{

        "fig_width":10.5,
        "fig_height":9.5,
        "x_axis_label":"",
        "y_axis_label":"Normalized Median Rank",
        "y_axis_label_fontsize":35,
        "y_axis_linewidth":2,
        "x_axis_linewidth":2, 
        "y_axis_range":(0, 1),
        "xticks_positions":[0, 1, 2, 3, 4, 5, 6],
        "x_tick_pointer_width":4,
        "y_tick_pointer_width":4,
        "xticks_labels":[],
        "xticks_fontsize":30,
        "xticks_rotation":45,
        "yticks_fontsize":30

    },

    "tSNE_original":{
        
        "file_type":"pdf",
        "fig_width":12,
        "fig_height":8,
        "num_components":2,
        "init_method":"random",
        "learning_rate":"auto",
        "point_size_s":0.5,
        "tick_fonts":14,
        "dist_from_top":1.25,
        "dist_from_bottom":0,
        "legend_title":"",
        "legend_title_font":16, 
        "legend_location":"center",
        "legend_position":(0.5, -0.2),
        "legend_fontsize":14,
        "legend_num_cols":11,
        "is_framed":False,
        "handleheight":1,
        "marker_scale":8,
        "column_spacing":0.5,
        "label_spacing":0.5,
        "test_size":0.3

    },

    "tSNE_embs":{

        "file_type":"pdf",
        "fig_width":12,
        "fig_height":8,
        "num_components":2,
        "init_method":"random",
        "learning_rate":"auto",
        "tick_fonts":14,
        "dist_from_top":1.25,
        "dist_from_bottom":0,
        "handleheight":1,
        "marker_scale":8,
        "column_spacing":0.5,
        "label_spacing":0.5

    },

    "all_plots":{

        "fig_width":"",
        "fig_height":"",
        "horizontal_space":0.3,
        "vertical_space":0.3,
        

    }
}