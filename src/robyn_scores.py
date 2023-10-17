from ssm_model.scores import print_scores
import numpy as np
from pathlib import Path
from pandas import read_csv
from ssm_model.constants import ModelSpecifications as ms
from constants import RobynModels

# Gets the scores for the train and test sets for the found robyn models
for model_number in RobynModels.MODELS:
    location = Path(
        f"/Users/jakob/Documents/ACMetric/MMM_article/results/models/RobynModel-{model_number}_predictions.csv"
    )
    df = read_csv(location)

    # Get the size of the train_set
    trainsize = int(round(df["TRAINSIZE"].iloc[0] * len(df)))

    x_train = df.iloc[:trainsize]
    x_test = df.iloc[trainsize:]

    # print("===== Train =====")
    # print_scores(
    #     x_train["OBSERVATIONS"].to_numpy(),
    #     x_train["PREDICTION"].to_numpy(),
    #     ms.ESTIMATED_PARAMTERS,
    # )
    print(f"\n===== Test {model_number} =====")
    print_scores(
        x_test["OBSERVATIONS"].to_numpy(),
        x_test["PREDICTION"].to_numpy(),
        ms.ESTIMATED_PARAMTERS,
    )
    # print_scores(true, prediction, penalty)
