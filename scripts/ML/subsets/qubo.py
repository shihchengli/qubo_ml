import pandas as pd
import torch.nn as nn
from qubo_ml.cross_validate import cross_validate

from scripts.ML.plot import plot_parity

data_path = "../../../data/cn-processed/subsets"
for split in ["s1", "s3", "s7", "s9"]:
    save_dir = split
    cross_validate(
        path=f"{data_path}/{split}/{split}_all.csv",
        seed=0,
        pytorch_seed=0,
        split_type="random",
        split_key_molecule=0,
        num_folds=1,
        ensemble_size=5,
        batch_size=5,
        epochs=200,
        lr=1e-3,
        task_names=["yield"],
        num_tasks=1,
        features_path=[f"{data_path}/{split}/{split}_all_features.csv"],
        smiles_columns=["substrate_smiles"],
        features_generator=None,
        save_splits=True,
        num_workers=0,
        no_cuda=False,
        gpu=None,
        loss_function=nn.MSELoss(reduction="none"),
        main_metric="rmse",
        metrics=["rmse", "mae", "r2"],
        save_dir=save_dir,
        save_preds=True,
        quiet=False,
    )

    # Plot figure
    y_true = pd.read_csv(f"{save_dir}/fold_0/test_full.csv")["yield"].tolist()
    y_pred = pd.read_csv(f"{save_dir}/fold_0/test_preds.csv")["yield"].tolist()
    file_name = f"{save_dir}/parity_plot.svg"
    plot_parity(y_true, y_pred, title=split, file_name=file_name)
