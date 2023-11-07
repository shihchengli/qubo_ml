import re
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def plot_learning_curve(log_path):
    """
    Plot the learning curve based on the validation error from a log file.

    Args:
        log_path (str): Path to the log file.

    Returns:
        None
    """
    # Read the file
    with open(log_path, 'r') as file:
        content = file.read()

    # Only collect the information from the final run
    runs = content.split("Loading data")
    last_run = runs[-1]

    # Use regex to find the numbers after "Validation rmse"
    pattern = r'Validation rmse = (\d+\.\d+)'
    matches = re.findall(pattern, last_run)

    # Save the validation error to a list
    validation_error_list = []
    for match in matches:
        validation_error_list.append(float(match))

    # Plot the learning curve
    plt.figure(dpi=100)
    plt.plot(validation_error_list)
    plt.xlabel('Number of epochs')
    plt.ylabel('Validation rmse')

    # Mark the model with the lowest validation error
    lowest_value = min(validation_error_list)
    lowest_index = validation_error_list.index(lowest_value)
    plt.scatter(lowest_index, lowest_value, color='red', s=100, facecolors='none')

def plot_parity(y_true, y_pred, title=None, y_pred_unc=None, file_name=None, error_bar=False):

    plt.figure(dpi=100)
    axmin = min(min(y_true), min(y_pred)) - 0.1*(max(y_true)-min(y_true))
    axmax = max(max(y_true), max(y_pred)) + 0.1*(max(y_true)-min(y_true))

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    plt.plot([axmin, axmax], [axmin, axmax], '--k')
    if error_bar:
        plt.plot([axmin-1, axmax-1], [axmin, axmax], '--r')
        plt.plot([axmin+1, axmax+1], [axmin, axmax], '--r')

    plt.errorbar(y_true, y_pred, yerr=y_pred_unc, linewidth=0, marker='o', markeredgecolor='black', markersize=6, alpha=0.5, elinewidth=1)

    plt.xlim((axmin, axmax))
    plt.ylim((axmin, axmax))

    ax = plt.gca()
    ax.set_aspect('equal')

    # if mae_std and rmse_std:
    #     at = AnchoredText(
    #     f"MAE = {mae:.3f} +/- {mae_std:.3f}\nRMSE = {rmse:.3f} +/- {rmse_std:.3f}", prop=dict(size=10), frameon=True, loc='upper left')
    #     at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    # else:
    #     at = AnchoredText(
    #     f"MAE = {mae:.3f}\nRMSE = {rmse:.3f}", prop=dict(size=10), frameon=True, loc='upper left')
    #     at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    # ax.add_artist(at)

    # Add a text box with MAE and RMSE values
    at = AnchoredText(
    f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}\nr2_score = {r2:.2f}", prop=dict(size=10), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

    plt.xlabel('Experimental yield (%)')
    plt.ylabel('Predicted yield (%)')

    if title:
      plt.title(title)

    if file_name:
        plt.tight_layout()
        plt.savefig(file_name, dpi=500, transparent=True)
    # plt.show()
    return

def plot_parity_train_test(y_train_true, y_train_pred, y_test_true, y_test_pred, title=None, y_train_pred_unc=None, y_test_pred_unc=None, file_name=None):

    plt.figure(dpi=100)
    y_true = y_train_true + y_test_true
    y_pred = y_train_pred + y_test_pred

    axmin = min(min(y_true), min(y_pred)) - 0.1*(max(y_true)-min(y_true))
    axmax = max(max(y_true), max(y_pred)) + 0.1*(max(y_true)-min(y_true))

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    plt.plot([axmin, axmax], [axmin, axmax], '--k')

    plt.errorbar(y_test_true, y_test_pred, yerr=y_test_pred_unc, linewidth=0, marker='o', markeredgecolor='black', markersize=6, alpha=0.5, elinewidth=1, label="test")
    plt.errorbar(y_train_true, y_train_pred, yerr=y_train_pred_unc, linewidth=0, marker='o', markeredgecolor='black', markersize=6, alpha=0.5, elinewidth=1, label="train")
    #plt.errorbar(y_test_true, y_test_pred, yerr=y_test_pred_unc, linewidth=0, marker='o', markeredgecolor='black', markersize=6, alpha=0.5, elinewidth=1, label="test")

    plt.xlim((axmin, axmax))
    plt.ylim((axmin, axmax))

    ax = plt.gca()
    ax.set_aspect('equal')

    at = AnchoredText(
    f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}\nr2_score = {r2:.2f}", prop=dict(size=10), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

    plt.xlabel('Experimental yield (%)')
    plt.ylabel('Predicted yield (%)')

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,0]
    #plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower right')

    if title:
      plt.title(title)

    if file_name:
        plt.tight_layout()
        plt.savefig(file_name, dpi=500, transparent=True)
    # plt.show()
    return