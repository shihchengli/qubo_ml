# predict

import csv

import numpy
from sklearn.metrics import mean_absolute_error, mean_squared_error


# This function read the features and yields from input csv files and output them as List.
def read_csv(filename):
    with open(filename, "r") as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        id_yield = header.index("yield")
        yield header
        for row in csv_reader:
            row[id_yield] = float(row[id_yield]) * 10  # yield*10
            yield row[-35:] + [row[id_yield]]  # [features*35 + yield]


# Read the testing dataset csv file
title = argv[1]
test_filename = title + "/test_features.csv"
inputs = read_csv(test_filename)
header = next(inputs)[-35:]
inputs = list(inputs)
real = [input[-1] / 10 for input in inputs]

# Find the reactions with top 20 real yield rates.
real_ids = list(numpy.argsort(real))
real_ids.reverse()
top_real = [real[real_id] for real_id in real_ids[:20]]

# Initialize the performance metrices
mae_all = []
mse_all = []
top_5 = []
top_10 = []
top_15 = []
top_20 = []
two_row_only = True  # If True, only the real yield and prediction yield outputted. If False, the features are also outputted.

# Analyze all the 5 results from DAU
for i in range(5):
    # Read the resulting contributions for each feature and feature pairs
    coefficient_file = title + "/" + title + "_coefficient_" + str(i) + ".csv"
    with open(coefficient_file, "r") as f:
        reader = csv.reader(f)
        c_header = next(reader)
        coefficient = {rows[0]: int(rows[1]) for rows in reader}

    # Output yield prediction csv file
    predict = []
    output_file = "predict_" + title + "_" + str(i) + ".csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        if two_row_only:
            writer.writerow(["yield"] + ["prediction"])
        else:
            writer.writerow(["yield"] + ["prediction"] + header[:-1])
        for input in inputs:
            real_yield = input[-1] / 10  # the last column in input data is yield
            prediction = 0
            feature_ids = [
                feature_id for feature_id, value in enumerate(input) if value == "1"
            ]
            for id1 in feature_ids:
                condition1 = header[id1]
                prediction += coefficient.get(condition1, 512) - 512  # linear terms
                id2s = [id2 for id2 in feature_ids if id2 > id1]
                for id2 in id2s:
                    condition2 = header[id2]
                    prediction += (
                        coefficient.get(str((condition1, condition2)), 512) - 512
                    )  # quadratic terms
            prediction /= 10
            predict.append(prediction)

        sorted_ids = list(numpy.argsort(predict))
        sorted_ids.reverse()
        for sorted_id in sorted_ids:
            if two_row_only:
                writer.writerow([inputs[sorted_id][-1] / 10] + [predict[sorted_id]])
            else:
                writer.writerow(
                    [inputs[sorted_id][-1] / 10]
                    + [predict[sorted_id]]
                    + inputs[sorted_id][:-1]
                )

        # Caluculate top_k accuracy
        top5 = 0
        for id_5 in sorted_ids[:5]:
            if real[id_5] in top_real[:5]:
                top5 += 1
        # writer.writerow(['top 5', top5])
        top_5.append(top5)

        top10 = 0
        for id_10 in sorted_ids[:10]:
            if real[id_10] in top_real[:10]:
                top10 += 1
        # writer.writerow(['top 10', top10])
        top_10.append(top10)

        top15 = 0
        for id_15 in sorted_ids[:15]:
            if real[id_15] in top_real[:15]:
                top15 += 1
        # writer.writerow(['top 15', top15])
        top_15.append(top15)

        top20 = 0
        for id_20 in sorted_ids[:20]:
            if real[id_20] in top_real[:20]:
                top20 += 1
        # writer.writerow(['top 20', top20])
        top_20.append(top20)

        # Calculate mean absolute error and mean square error
        mae = mean_absolute_error(real, predict)
        mse = mean_squared_error(real, predict)
        # writer.writerow(['mean absolute error', mae])
        # writer.writerow(['mean squared error', mse])
        mae_all.append(mae)
        mse_all.append(mse)

# Check the average performance
print(
    "top 5,10,15,20:", sum(top_5) / 5, sum(top_10) / 5, sum(top_15) / 5, sum(top_20) / 5
)
print("mae, mse:", sum(mae_all) / 5, sum(mse_all) / 5)
