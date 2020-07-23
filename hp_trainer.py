from src.options import get_opts_combs
from src.train import train

opts = {
    # ---folders---
    "experiment_name": ['fc_imagemethod_lowrank'],
    # ---network structure---
    "model_name": ['fc'],
    "rank": [10,25,35],  # None -> output is full matrix, Int -> output is low rank matrix transformed into full matrix
    "hidden_layers": [1,2],
    "hidden_sizes": [1000, 2000],
    "residual_flag": [True,False],
    "residual_only": [False],
    "loss": ['mse'],  # 'mse'
    # ---data---
    # "dtype": torch.float32, # TODO: implement (does errors in saving hyperparameters)
    "transform": [None],
    "batch_size": [10, 25, 50, 100],
    "num_workers": [8],
    "train_val_split": [[0.9, 0.1]],
    "preload_data": [True],
    # ---optimization---
    "lr": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    "max_epochs": [100],
    "gpus": [-1]
}


def optimize_hp(opts):
    opts_list = get_opts_combs(opts)
    for opts in opts_list:
        train(opts)


if __name__ == '__main__':
    optimize_hp(opts)
