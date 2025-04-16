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


# model_order = ["sCIN", "Con-AAE", "scBridge", "scGLUE", "MOFA", "Autoencoder", "Harmony"]
model_order = ["Paired", "50%", "20%", "10%", "5%", "1%", "Random"]


model_palette = {
    "Con-AAE": "#e41a1c",
    "sCIN": "#377eb8",
    "scBridge": "#4daf4a",
    "scGLUE":"#984ea3",
    "MOFA": "#ff7f00",
    "Autoencoder":"#a65628",        
    "Harmony": "#a6cee3"
}


plots = {
    "recall_at_k":{

        "file_type":"png",
        "fig_width":8,
        "fig_height":6,
        "err_bar_format":"-",
        "err_bar_capsize":5,
        "err_bar_capthick":1,
        "err_bar_linewidth":2,
        "legend_names":{
            "sCIN":"sCIN",
            "Con-AAE":"Con-AAE",
            "MOFA":"MOFA",
            "Harmony":"Harmony",
            "Autoencoder":"Autoencoder",
            "scBridge":"scBridge",
            "scGLUE":"scGLUE"
        },
        "legend_location":"lower center",
        "legend_position":(0.365, 0.68),
        "legend_title":"",
        "legend_num_cols":2,
        "legend_fontsize":18,
        "legend_frame":False,
        "legend_linewidth":4,
        "x_axis_fontsize":20,
        "y_axis_fontsize":20,
        "xticks_positions":[10, 20, 30, 40, 50],
        "xticks_fontsize":18,
        "yticks_fontsize":18,
        "y_axis_range":(0, 0.40)

    },

    "ASW":{

        "file_type":"png",
        "fig_width":8,
        "fig_height":6,
        "x_axis_label":"",
        "y_axis_label":"ASW",
        "y_axis_label_fontsize":20,
        "y_label_fontsize":20,
        "xticks_positions":[0, 1, 2, 3, 4, 5, 6],
        "xticks_labels":['Con-AAE', 'sCIN', 'scBridge', 'MOFA', 'scGLUE', 'AE', 'Harmony'],
        "xticks_fontsize":18,
        "xticks_rotation":45,
        "yticks_fontsize":18,
        "y_axis_range":(0, 0.8)

    },

    "cell_type_accuracy":{

        "file_type":"png",
        "fig_width":8,
        "fig_height":6,
        "x_axis_label":"",
        "y_axis_label":"Cell Type Accuracy",
        "y_axis_label_fontsize":20,
        "y_axis_range":(0, 0.8),
        "xticks_positions":[0, 1, 2, 3, 4, 5, 6],
        "xticks_labels":["sCIN", "Con-AAE", "Harmony", "MOFA", "Autoencoder", "scBridge", "scGLUE"],
        "xticks_fontsize":18,
        "xticks_rotation":45,
        "yticks_fontsize":18
        
    },

    "cell_type_accuracy_joint":{

        "file_type":"png",
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

        "file_type":"png",
        "fig_width":8,
        "fig_height":6,
        "x_axis_label":"",
        "y_axis_label":"Normalized Median Rank",
        "y_axis_label_fontsize":20,
        "y_axis_range":(0, 0.8),
        "xticks_positions":[0, 1, 2, 3, 4, 5, 6],
        "xticks_labels":[],
        "xticks_fontsize":18,
        "xticks_rotation":45,
        "yticks_fontsize":18

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