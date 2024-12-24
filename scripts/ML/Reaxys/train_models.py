import os

import pandas as pd
import torch.nn as nn
from qubo_ml.cross_validate import cross_validate

from scripts.ML.plot import plot_parity

data_set = "buchwald"  # change this to the other dataset (e.g., negishi, suzuki)
features_generators = ["MACCS", "Avalon", "ECFP4"]
reaction_modes = [
    "reac_prod",
    "reac_diff",
    "prod_diff",
    "reac_only",
    "prod_only",
    "diff_only",
]
for features_generator, num_bits in [("MACCS", 166), ("Avalon", 1024), ("ECFP4", 2048)]:
    for reaction_mode in reaction_modes:
        data_path = f"/Path/to/data_path/{data_set}"
        save_dir = (
            f"/Path/to/svae_dir/{data_set}/test/{features_generator}/{reaction_mode}"
        )
        if os.path.exists(save_dir):
            continue
        cross_validate(
            path=f"{data_path}/filtered_{data_set}.csv",
            seed=0,
            pytorch_seed=0,
            split_type="random",
            split_key_molecule=0,
            num_folds=1,
            ensemble_size=5,
            batch_size=50,
            epochs=200,
            lr=1e-3,
            task_names=["Yield (numerical)"],
            num_tasks=1,
            features_path=[
                f"{data_path}/filtered_{data_set}_reagent.csv",
                f"{data_path}/filtered_{data_set}_sol.csv",
                f"{data_path}/filtered_{data_set}_temp.csv",  # Remove for no temperature info, or use `filtered_{data_set}_binary_temp.csv` for binary encoding.
            ],
            smiles_columns=["Reaction"],
            features_generator=[features_generator],
            num_bits=num_bits,
            save_splits=True,
            num_workers=0,
            no_cuda=False,
            gpu=None,
            loss_function=nn.MSELoss(reduction="none"),
            main_metric="rmse",
            metrics=["rmse", "mae", "r2"],
            save_dir=save_dir,
            save_preds=True,
            is_reaction=True,
            reaction_mode=reaction_mode,
            quiet=False,
        )

        # Plot figure
        y_true = pd.read_csv(f"{save_dir}/fold_0/test_full.csv")[
            "Yield (numerical)"
        ].tolist()
        y_pred = pd.read_csv(f"{save_dir}/fold_0/test_preds.csv")[
            "Yield (numerical)"
        ].tolist()
        file_name = f"{save_dir}/parity_plot.svg"
        plot_parity(
            y_true,
            y_pred,
            title=f"{features_generator}({reaction_mode})",
            file_name=file_name,
        )
