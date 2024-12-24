import os

import pandas as pd
import torch.nn as nn
from qubo_ml.cross_validate import cross_validate

from scripts.ML.plot import plot_parity

data_path = "../../../../data"

features_set = [
    "*Cl",
    "*Br",
    "*I",
    "*OC",
    "*CC",
    "*c1ccccn1",
    "*c1cccnc1",
    "*C(F)(F)F",
    "P2Et",
    "BTMG",
    "MTBD",
    "XPhos",
    "tBuXPhos",
    "tBuBrettPhos",
    "AdBrettPhos",
    "a1",
    "a2",
    "a3",
    "a4",
    "a5",
    "a6",
    "a7",
    "a8",
    "a9",
    "a10",
    "a11",
    "a12",
    "a13",
    "a14",
    "a15",
    "a16",
    "a17",
    "a18",
    "a19",
    "a20",
]

all_sets = ["s1", "s3", "s7", "s9"]
for i in range(len(all_sets)):
    test_set = [all_sets[i]]
    # train_set = all_sets[:i] + all_sets[i+1:]
    train_set = [f"s{i+1}" for i in range(15)]
    train_set.remove(all_sets[i])

    save_dir = "./" + "".join(train_set) + "_" + test_set[0] + "_qubo"
    os.makedirs(save_dir, exist_ok=True)

    # Prepare data
    df = pd.read_csv(
        f"{data_path}/whole_dataset/cn-processed_features.csv", index_col=0
    )

    # train set
    train_df = df[df["substrate_id"].isin(train_set)]
    train_df.to_csv(f"{save_dir}/train.csv")
    train_df[features_set].to_csv(f"{save_dir}/train_features.csv", index=False)

    # test set
    test_df = df[df["substrate_id"].isin(test_set)]
    test_df.to_csv(f"{save_dir}/test.csv")
    test_df[features_set].to_csv(f"{save_dir}/test_features.csv", index=False)

    # Run the model
    cross_validate(
        path=f"{save_dir}/train.csv",
        separate_test_path=f"{save_dir}/test.csv",
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
        features_path=[f"{save_dir}/train_features.csv"],
        separate_test_features_path=[f"{save_dir}/test_features.csv"],
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
    plot_parity(y_true, y_pred, title=test_set[0], file_name=file_name)
