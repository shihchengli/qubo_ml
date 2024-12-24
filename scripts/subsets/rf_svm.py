import chemprop
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.offsetbox import AnchoredText
from qubo_ml.sklearn_train import run_sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mpl.use("tkagg")


def plot_parity(
    y_true, y_pred, title=None, y_pred_unc=None, file_name=None, error_bar=False
):
    plt.figure(dpi=100)
    axmin = min(min(y_true), min(y_pred)) - 0.1 * (max(y_true) - min(y_true))
    axmax = max(max(y_true), max(y_pred)) + 0.1 * (max(y_true) - min(y_true))

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

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
        f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}\nr2_score = {r2:.2f}",
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
    # plt.show()
    return


data_path = "../../data/cn-processed/subsets"
for model_type in ["random_forest", "svm"]:
    for split in ["s1", "s3", "s7", "s9"]:
        save_dir = f"{model_type}/{split}"
        arguments = [
            "--data_path",
            f"{data_path}/{split}/{split}_all.csv",
            "--dataset_type",
            "regression",
            "--seed",
            "0",
            "--num_folds",
            "1",
            "--split_type",
            "random",
            "--target_columns",
            "yield",
            "--features_path",
            f"{data_path}/{split}/{split}_all_features.csv",
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
            "--model_type",
            model_type,
        ]
        print("arguments:", arguments)
        args = chemprop.args.SklearnTrainArgs().parse_args(arguments)
        mean_score, std_oscore = chemprop.train.cross_validate(
            args=args, train_func=run_sklearn
        )

        # Plot figure
        y_true = pd.read_csv(f"{save_dir}/fold_0/test_full.csv")["yield"].tolist()
        y_pred = pd.read_csv(f"{save_dir}/fold_0/test_preds.csv")["yield"].tolist()
        file_name = f"{save_dir}/parity_plot.svg"
        plot_parity(y_true, y_pred, title=split, file_name=file_name)
