from numpy import (
    empty,
    nan,
    ndarray,
    matrix,
    zeros,
    pi,
    asarray,
    sin,
    cos,
    array,
    sum,
    zeros,
    sort,
)
from scipy.signal import lfilter
from time import perf_counter
from logging import Logger
from ssm_model.constants import Optimization as op
from pandas import DataFrame, concat, to_datetime, read_csv, options, Series
from constants import SaveLocation as sl
import csv
from ssm_model.estimated_parameters import EstimatedParameters
from pathlib import Path
from constants import RobynModels
from data_generation.dg_constants import DataHelpers
from constants import SaveLocation as sl

options.mode.chained_assignment = None  # default='warn'


def empty_vector(size: int) -> ndarray:
    """Make an empty vector of a specific size

    Args:
        size (int): How big your vector will be

    Returns:
        ndarray: The vector of a specific size
    """
    vector = empty(size)
    vector.fill(nan)
    return vector


def empty_3d_array(time_steps: int, length: int, width: int) -> matrix:
    """Make an empty 3d matrix of a specific size

    Args:
        size (int): How big your vector will be

    Returns:
        ndarray: The vector of a specific size
    """
    matr = zeros((time_steps, length, width))
    matr.fill(nan)
    return matr


def lamb(s: int, j: int) -> float:
    return (2 * pi * j) / s


def diagonally_append(*matrices: ndarray) -> matrix:
    """Diagonally append matrices where the off diagonals are all zeros

    Returns:
        matrix: the combined matrixes
    """
    # Define the shape of the result array
    shape = sum([matrix.shape for matrix in matrices], axis=0)
    result = zeros(shape)

    # Define the indices for placing each array
    idx1 = idx2 = 0
    for matrix in matrices:
        rows, cols = matrix.shape
        result[idx1 : idx1 + rows, idx2 : idx2 + cols] = matrix
        idx1 += rows
        idx2 += cols

    return result


def C_matrix(lamb: float) -> matrix:
    """Returns the C matrix is described in 3.11 page 48 of Koopman and Durbin

    Args:
        lambd (float): the value you put into the sin and cos

    Returns:
        matrix: The C matrix
    """
    return array([[cos(lamb), sin(lamb)], [-sin(lamb), cos(lamb)]])


def apply_marketing_transformation(
    x_raw: ndarray, theta: float, phi: float, rho: float
) -> ndarray:
    """Applies the adstock and saturation transformation on the raw marketing data"""

    x_adstocked = apply_adstock(x_raw, theta)
    # Scale phi by multiplying it by the max in value in the array
    return apply_saturation(x_adstocked, phi, rho * (x_adstocked.max()))


def apply_marketing_transformation_single(
    x_raw: ndarray,
    x_single: float,
    theta: float,
    phi: float,
    rho: float,
) -> ndarray:
    """Applies the adstock and saturation transformation on the raw marketing data"""

    x_adstocked = apply_adstock(x_raw, theta)
    # Scale phi by multiplying it by the max in value in the array
    return apply_saturation(x_single, phi, rho * (x_adstocked.max()))


def apply_adstock(x_raw: ndarray, theta: float) -> ndarray:
    """Applies adstocked transformation on the raw marketing data"""
    return lfilter([1], [1, -theta], x_raw)


def apply_saturation(x_adstocked: ndarray, phi: float, rho: float) -> ndarray:
    """Applies saturation transformation on the adstocked data
    phi= alpha, rho= gamma in other sources"""
    return x_adstocked**phi / (x_adstocked**phi + rho**phi)


def save_to_csv(parameters_list: list, save_location: Path):
    """Converts the list of objects to a CSV and save it locally

    Args:
        parameters_list (list): The list with objects
        save_location (Path): Where the CSV will be located
    """
    # Open the CSV file
    with open(save_location, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write the header (keys of the dictionary) if the list is not empty
        if parameters_list:
            header = parameters_list[0].__dict__.keys()
            writer.writerow(header)

            # Write the values (items of the dictionary) for each object
            for parameters in parameters_list:
                writer.writerow(parameters.__dict__.values())


def csv_to_list(save_location: Path) -> list:
    """Opens the saved estimated parameters CSV and converts it to a list

    Args:
        save_location (Path): Location where your CSV is

    Returns:
        list: The list with each entry being a run of the model
    """
    parameters_list = []

    # Read the CSV file
    with open(save_location, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip the header row

        # Convert each row to an EstimatedParameters object
        for row in reader:
            ep = EstimatedParameters(False)
            ep.set_via_array(row)
            parameters_list.append(ep)

    return parameters_list


def save_dataset(marketing_data: DataFrame, y: ndarray, random_state: int):
    """Save the generated data into a csv, also save the promo separately because that is needed in Robyn

    Args:
        marketing_data (DataFrame): Marketing and promo variables
        y (ndarray): The observations/ revenue
        random_state (int): Random state used for generating
    """
    save_location_dataset = sl.ROBYN_100RUNS / f"dataset_(state={random_state}).csv"
    save_location_promo = sl.ROBYN_100RUNS / f"promo.csv"

    marketing_data["Revenue"] = y
    marketing_data.to_csv(save_location_dataset)

    promo = get_promo_df(marketing_data)
    promo.to_csv(save_location_promo, index=False)


def get_promo_df(marketing_data: DataFrame) -> DataFrame:
    # Convert the columns "Promo_1" and "Promo_2" into rows
    df_promo_1 = marketing_data[marketing_data["Promo_1"] == 1].copy()
    df_promo_2 = marketing_data[marketing_data["Promo_2"] == 1].copy()

    df_promo_1["holiday"] = "Promo_1"
    df_promo_2["holiday"] = "Promo_2"

    # Concatenate the two DataFrames
    df_promos_new = concat([df_promo_1, df_promo_2], axis=0)

    # Add the "Country" column with the value "CL" for all rows
    df_promos_new["country"] = "CL"

    # Extract the year from the "Date" column and create a "Year" column
    df_promos_new["year"] = to_datetime(df_promos_new["Date"]).dt.year

    # Reorder and select columns
    df_promos_new = df_promos_new[["Date", "holiday", "country", "year"]]
    df_promos_new.rename(columns={"Date": "ds"}, inplace=True)

    # Show the resulting DataFrame
    return df_promos_new


def get_robyn_parameters(model_numbers: list[int]) -> DataFrame:
    """Get the parameters for each robyn model into one dataframe

    Args:
        model_numbers (list[int]): Models you want to retrieve the parameters for

    Returns:
        DataFrame: Dataframe containing all the parameters per model, each row is one model
    """
    frames = []

    for model_number in model_numbers:
        robyn_model_location = (
            sl.ROBYN_MODELS / f"RobynModel-{model_number}_parameters.csv"
        )
        df = read_csv(robyn_model_location)
        df["model_number"] = model_number
        frames.append(df)

    # Concatenate the list of dataframes into one large dataframe
    big_df = concat(frames, ignore_index=True)
    return big_df


def get_robyn_predictions(model_numbers: list[int]) -> list[DataFrame]:
    """Convert the R-csvs into a list of dataframes. One dataframe per model

    Args:
        model_numbers (list[int]): Models you want to convert to dataframes

    Returns:
        list[DataFrame]: List of the different predictions per model
    """
    dfs = []

    for model_number in model_numbers:
        robyn_model_location = (
            sl.ROBYN_MODELS / f"RobynModel-{model_number}_predictions.csv"
        )
        df = read_csv(robyn_model_location)

        # Add all the marketing channel effects into one new columns
        df["MARKETING"] = df[
            [
                "CHANNEL_1",
                "CHANNEL_2",
                "CHANNEL_3",
                "CHANNEL_4",
                "CHANNEL_5",
                "CHANNEL_6",
            ]
        ].sum(axis=1)
        df["MODEL_NUMBER"] = model_number

        dfs.append(df)

    return dfs


def make_column_names(robyn_models: list[str]) -> list[str]:
    column_names = ["Date", "Original", "Optimal", "Optimal_perc"]
    for model in RobynModels.MODELS:
        column_names.append(f"{model}")
        column_names.append(f"{model}_perc")
    return column_names


def combine_robyn_dfs(
    robyn_hyperparam_df: DataFrame, robyn_beta_df: DataFrame, model_number
) -> DataFrame:
    combined_df = robyn_hyperparam_df.loc[robyn_hyperparam_df["solID"] == model_number]

    # For each channle extract the coefficient (the value of beta)
    for channel in DataHelpers.FAKE_CHANNELS:
        combined_df[channel] = robyn_beta_df.loc[
            (robyn_beta_df["solID"] == model_number) & (robyn_beta_df["rn"] == channel)
        ]["coef"].values[0]

    return combined_df
