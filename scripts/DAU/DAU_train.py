# This code read yield datasets, construct QUBO terms, call DAU's API, get the annealing results, and output contribution of each structural features for prediction.

import csv
import json
import sys
import time

import requests

# DAU authority setting
API_Key = "Key"
API_Access_URL = "URL"
headers = {"content-type": "application/json", "X-Api-Key": API_Key}
proxies = {}


# Expand the contribution for every feature and feature pair found in the reaction
# The contributions look like this: x_0 + 2x_1 + 4x_2 + ... _ 512x_9 - 512
# For the features with '1' in a reaction, the variable ids for linear terms and quadratic terms are found in the coding Dictionary.
def expand_binary_qubo(input, coding):
    bp = {}  # key=variable, value=coefficient
    feature_ids = [feature_id for feature_id, value in enumerate(input) if value == "1"]
    for feature_id1 in feature_ids:
        for i in range(10):  # 1, 2 ,4, 8, ... , 512    #linear term
            bp[feature_id1 * 10 + i] = 2**i
        feature_ids2 = [
            feature_id2 for feature_id2 in feature_ids if feature_id2 > feature_id1
        ]
        for feature_id2 in feature_ids2:  # quadratic term
            condition1 = coding[feature_id1]
            condition2 = coding[feature_id2]
            base_key = coding.index((condition1, condition2)) * 10
            for i in range(10):
                bp[base_key + i] = 2**i
    return bp


# This function calculates Equation (7) of single reaction: (P_i-Y_i)^2
# P_i is the prediction yield of reaction i, which corresponds to Equation (6), composing of the contributions of features from linear terms and quadratic terms
# The contribution part is calculated in expand_binary_qubo().
# Y_i is the yield of reaction i.
def qubo_square(input, coding):
    reaction_yield = input[-1]
    variables = expand_binary_qubo(input, coding)
    n_feature = len(variables) / 10
    output = variables.copy()
    # linear terms
    for key in variables.keys():
        output[key] = (variables[key] ** 2) - 2 * variables[key] * (
            reaction_yield + 512 * n_feature
        )
    # quadratic terms
    keys = list(variables.keys())
    keys.sort()
    n_keys = len(keys)
    for id1 in range(n_keys):
        for id2 in range(id1 + 1, n_keys):
            output[(keys[id1], keys[id2])] = (
                2 * variables[keys[id1]] * variables[keys[id2]]
            )
    output["constant"] = (reaction_yield + 512 * n_feature) ** 2
    return output


# Sum all the terms from different reactions
def merge_qubo(qubo_list):
    output = {}
    for qubo in qubo_list:
        for key in qubo.keys():
            output[key] = output.get(key, 0) + qubo[key]
    return output


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


# Transform the QUBO terms into the format requested by DAU API.
def dau_encode(objective_function, penalty_function):
    bp_terms = [{"c": objective_function["constant"], "p": []}]
    for term, coefficient in objective_function.items():
        if isinstance(term, int):
            bp_terms.append({"c": coefficient, "p": [term]})
        elif isinstance(term, tuple):
            bp_terms.append({"c": coefficient, "p": list(term)})
    bp = {"terms": bp_terms}
    if penalty_function:
        pbp_terms = [{"c": penalty_function["constant"], "p": []}]
        for term in penalty_function:
            if isinstance(term, int):
                pbp_terms.append({"c": penalty_function[term], "p": [int(term)]})
            elif isinstance(term, tuple):
                pbp_terms.append({"c": penalty_function[term], "p": list(term)})
        pbp = {"terms": pbp_terms}
    else:
        pbp = None
    return (bp, pbp)


# Be cautious! The required scheme for training set is different from the original file, but can be generated by concatenating two related files.
# The training, validation, and testing datasets are csv files composing of 37 columns. (C-N cross coupling dataset)
# Column 1: Substrate_SMILES
# Column 2: Yield
# Column 3-37: Features

# Define the input files
# For single substrate predictions, all the datasets come from the same file with the same title. Eg. title = 's1'
# To predict the yield of one substrates from three other substrates, The training and validation datasets are concatenated, respectively.
# Eg. To predict s9 from s1, s3, and s7 datasets, training datasets of s1, s3, s7 are concatenated to form a new csv file, and so do the validation sets.

title = sys.argv[1]
test_filename = title + "/" + "test_features.csv"
train_filename = title + "/" + "train_features.csv"
val_filename = title + "/" + "val_features.csv"
response_filename = title + "/" + title + "_response.txt"

# Read the header line for feature names
val_inputs = read_csv(val_filename)
val_header = next(val_inputs)
coding = val_header[-35:]
print(coding)

# Build the mapping table for features according to the structural encoding
groups = {}
for condition_id, condition in enumerate(coding):
    if condition_id < 3:
        groups[condition] = 0  #'substrate_halide': 3
    elif condition_id < 8:
        groups[condition] = 1  #'substrate_variation': 5
    elif condition_id < 11:
        groups[condition] = 2  #'base': 3
    elif condition_id < 15:
        groups[condition] = 3  #'ligand': 4
    else:
        groups[condition] = 4  #'additive': 20

# Construct the quadratic terms
n_groups = len(set(groups.values()))  # =5
for group1 in range(n_groups):
    for group2 in range(group1 + 1, n_groups):
        groups1 = [condition for condition, group in groups.items() if group == group1]
        groups2 = [condition for condition, group in groups.items() if group == group2]
        for condition1 in groups1:
            for condition2 in groups2:
                coding.append(
                    (condition1, condition2)
                )  # 418 = 35 + 15 + 9 + 12 + 60 + 15 + 20 + 100 + 12 + 60 + 80

# Generate objective term that utilize training dataset
train_inputs = read_csv(train_filename)
header = next(train_inputs)
obj_qubo_list = []
for train_input in train_inputs:
    obj_qubo = qubo_square(train_input, coding)
    obj_qubo_list.append(obj_qubo)
objective_function = merge_qubo(obj_qubo_list)

# Generate penalty term that utilize validation dataset
pen_qubo_list = []
for val_input in val_inputs:
    pen_qubo = qubo_square(val_input, coding)
    pen_qubo_list.append(pen_qubo)
penalty_function = merge_qubo(pen_qubo_list)

# Encode the terms into the input form requested by DAU
bp, pbp = dau_encode(objective_function, penalty_function)

# Build the request to DAU server
request = {
    "fujitsuDA3": {
        # "time_limit_sec": int #Default=10
        # "num_output_solution": int #Default=5
        # "one_way_one_hot_groups": v_1w1h #None
        # "two_way_one_hot_groups": v_2w1h #None
    },
    "binary_polynomial": bp,
    "penalty_binary_polynomial": pbp,
    # "inequalities": [ineq1, ineq2, ...] #None
}

# Post request to DAU server
calling_API = requests.post(
    API_Access_URL + "/async/qubo/solve",
    json.dumps(request),
    headers=headers,
    proxies=proxies,
)
job_id = calling_API.json()["job_id"]
print(job_id)

# Wait for the results
time.sleep(60)

# Get the results
response = requests.get(
    API_Access_URL + "/async/jobs/result/" + job_id, headers=headers, proxies=proxies
)
# print(response.json())

# Record the results
with open(response_filename, "w", newline="") as f:
    print(response.json(), file=f)

delete = requests.delete(
    API_Access_URL + "/async/jobs/result/" + job_id, headers=headers, proxies=proxies
)

# Parse the results, calcuate and output the contribution of each feature into csv files
for i in range(5):
    solution = response.json()["qubo_solution"]["solutions"][i]["configuration"]
    coefficient = {}
    for id1, condition in enumerate(coding):
        if str(10 * id1) in solution:
            coefficient[condition] = 0
            for j in range(10):
                if solution.get(str(10 * id1 + j), False):
                    coefficient[condition] += 2**j
    output_filename = (
        title + "/" + title + "_coefficient_" + str(i) + "_" + version + ".csv"
    )
    with open(output_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variable_name", "coefficient", "yield (scale:*10)"])
        for key, value in coefficient.items():
            writer.writerow([key, value])
