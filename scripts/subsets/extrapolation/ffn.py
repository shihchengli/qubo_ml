import os

import chemprop
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import mean_absolute_error, mean_squared_error

mpl.use("tkagg")


def plot_parity(
    y_true, y_pred, title=None, y_pred_unc=None, file_name=None, error_bar=False
):
    plt.figure(dpi=100)
    axmin = min(min(y_true), min(y_pred)) - 0.1 * (max(y_true) - min(y_true))
    axmax = max(max(y_true), max(y_pred)) + 0.1 * (max(y_true) - min(y_true))

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    plt.plot([axmin, axmax], [axmin, axmax], "--k")
    if error_bar:
        plt.plot([axmin - 1, axmax - 1], [axmin, axmax], "--r")
        plt.plot([axmin + 1, axmax + 1], [axmin, axmax], "--r")
    plt.errorbar(
        y_true,
        y_pred,
        yerr=y_pred_unc,
        linewidth=0,
        marker="o",
        markeredgecolor="black",
        markersize=6,
        alpha=0.5,
        elinewidth=1,
    )

    plt.xlim((axmin, axmax))
    plt.ylim((axmin, axmax))

    ax = plt.gca()
    ax.set_aspect("equal")

    # Add a text box with MAE and RMSE values
    at = AnchoredText(
        f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}",
        prop=dict(size=10),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

    plt.xlabel("Experimental yield (%)")
    plt.ylabel("Predicted yield (%)")

    if title:
        plt.title(title)

    if file_name:
        plt.tight_layout()
        plt.savefig(file_name, dpi=500, transparent=True)
    return


data_path = "../../../data"

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
    train_set = all_sets[:i] + all_sets[i + 1 :]

    save_dir = "./" + "".join(train_set) + "_" + test_set[0] + "_ffn"
    os.makedirs(save_dir, exist_ok=True)

    # Prepare data
    df = pd.read_csv(f"{data_path}/cn-processed/cn-processed_features.csv", index_col=0)

    # train set
    train_df = df[df["substrate_id"].isin(train_set)]
    train_df.to_csv(f"{save_dir}/train.csv")
    train_df[features_set].to_csv(f"{save_dir}/train_features.csv", index=False)

    # test set
    test_df = df[df["substrate_id"].isin(test_set)]
    test_df.to_csv(f"{save_dir}/test.csv")
    test_df[features_set].to_csv(f"{save_dir}/test_features.csv", index=False)

    # Run the model
    arguments = [
        "--data_path",
        f"{save_dir}/train.csv",
        "--separate_test_path",
        f"{save_dir}/test.csv",
        "--dataset_type",
        "regression",
        "--seed",
        "0",
        "--pytorch_seed",
        "0",
        "--num_folds",
        "1",
        "--ensemble_size",
        "5",
        "--batch_size",
        "5",
        "--epochs",
        "50",
        "--init_lr",
        "1e-3",
        "--max_lr",
        "1e-3",
        "--final_lr",
        "1e-3",
        "--split_type",
        "random",
        "--target_columns",
        "yield",
        "--features_path",
        f"{save_dir}/train_features.csv",
        "--separate_test_features_path",
        f"{save_dir}/test_features.csv",
        "--smiles_columns",
        "substrate_smiles",
        "--metric",
        "rmse",
        "--extra_metrics",
        "mae",
        "r2",
        "--save_dir",
        save_dir,
        "--save_preds",
        "--save_smiles_splits",
        "--features_only",
    ]
    print("arguments:", arguments)
    args = chemprop.args.TrainArgs().parse_args(arguments)
    mean_score, std_score = chemprop.train.cross_validate(
        args=args, train_func=chemprop.train.run_training
    )

    # Plot figure
    y_true = pd.read_csv(f"{save_dir}/fold_0/test_full.csv")["yield"].tolist()
    y_pred = pd.read_csv(f"{save_dir}/fold_0/test_preds.csv")["yield"].tolist()
    file_name = f"{save_dir}/parity_plot.svg"
    plot_parity(y_true, y_pred, title=test_set[0], file_name=file_name)
