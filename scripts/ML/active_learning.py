import csv
import os
from random import Random

import pandas as pd
import torch.nn as nn
from qubo_ml.cross_validate import cross_validate

from scripts.ML.plot import plot_parity_train_test

data_path = "../data/cn-processed"
active_learning_methods = ["top_prediction", "random"]
n_data_added = 50

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

for active_learning_method in active_learning_methods:
    for n_fold in range(5):
        os.makedirs(f"{active_learning_method}/fold_{n_fold}/", exist_ok=True)
        f = open(
            os.path.join(f"{active_learning_method}/fold_{n_fold}/", "scores.csv"),
            "w",
            newline="",
        )
        writer = csv.writer(f)
        writer.writerow(
            [
                "Iterations",
                "Top-5 accuracy score",
                "Top-10 accuracy score",
                "Top-15 accuracy score",
                "Top-20 accuracy score",
            ]
        )

        train_df = pd.read_csv(
            f"{data_path}/random_100_data/fold_{n_fold}.csv", index_col=0
        )
        original_train_indeces = list(train_df.index)
        extra_train_indeces = []

        n_iter = 1
        while True:
            save_dir = f"{active_learning_method}/fold_{n_fold}/run_{n_iter}"
            os.makedirs(save_dir, exist_ok=True)

            # Prepare data
            df = pd.read_csv(f"{data_path}/cn-processed_features.csv", index_col=0)

            # train set
            train_indeces = original_train_indeces + extra_train_indeces
            train_df = df.loc[train_indeces]
            train_df.to_csv(f"{save_dir}/train.csv")
            train_df[features_set].to_csv(f"{save_dir}/train_features.csv", index=False)

            # test set
            separate_test_path = f"{data_path}/cn-processed_features.csv"
            separate_test_features_path = f"{data_path}/features.csv"

            # Run the model
            cross_validate(
                path=f"{save_dir}/train.csv",
                separate_test_path=separate_test_path,
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
                separate_test_features_path=[separate_test_features_path],
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
            train_df = pd.read_csv(f"{save_dir}/train.csv", index_col=0)
            train_indices = train_df.index
            test_df = pd.read_csv(separate_test_path, index_col=0)
            test_indices = []
            for i in range(len(test_df)):
                if i not in train_indices:
                    test_indices.append(i)

            y_train_true = (
                pd.read_csv(f"{save_dir}/fold_0/test_full.csv")
                .loc[train_indices]["yield"]
                .tolist()
            )
            y_train_pred = (
                pd.read_csv(f"{save_dir}/fold_0/test_preds.csv")
                .loc[train_indices]["yield"]
                .tolist()
            )
            y_test_true = (
                pd.read_csv(f"{save_dir}/fold_0/test_full.csv")
                .loc[test_indices]["yield"]
                .tolist()
            )
            y_test_pred = (
                pd.read_csv(f"{save_dir}/fold_0/test_preds.csv")
                .loc[test_indices]["yield"]
                .tolist()
            )

            file_name = f"{save_dir}/parity_plot.svg"
            plot_parity_train_test(
                y_train_true,
                y_train_pred,
                y_test_true,
                y_test_pred,
                title=f"Iteration: {n_iter}",
                file_name=file_name,
            )

            # Check stop criteria
            test_df_sorted = test_df.sort_values(by="yield")
            top_5_true_indices = test_df_sorted.index[-5:]

            pred_df = pd.read_csv(f"{save_dir}/fold_0/test_preds.csv")
            pred_df_sorted = pred_df.sort_values(by="yield")

            top_k_accuracy_scores = []
            for k in [5, 10, 15, 20]:
                top_k_true_indices = test_df_sorted.index[-k:]
                top_k_pred_indices = pred_df_sorted.index[-k:]
                intersection = set(top_k_true_indices) & set(top_k_pred_indices)
                top_k_accuracy_scores.append(len(intersection))

            writer.writerow([n_iter] + top_k_accuracy_scores)

            if n_iter == 10:  # len(test_indices) < n_data_added:
                print("Max iteration 10 is reached...")
                break

            n_iter += 1
            if active_learning_method == "top_prediction":
                test_preds_df = pd.read_csv(f"{save_dir}/fold_0/test_preds.csv").loc[
                    test_indices
                ]
                test_preds_df_sorted = test_preds_df.sort_values(by="yield")
                extra_train_indeces += list(test_preds_df_sorted.index[-n_data_added:])
            elif active_learning_method == "random":
                random = Random(0)
                random.shuffle(test_indices)
                extra_train_indeces += test_indices[:n_data_added]
            else:
                raise ValueError("Not supported active_learning_method.")
        f.close()
