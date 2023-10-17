from pathlib import Path
from ssm_model.estimated_parameters import EstimatedParameters
from time import perf_counter
from numpy import ndarray
import os
import csv
from pathlib import Path


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


def timing_run(tik: float, tik_tik: float, index: int):
    """Returns information on the duration of the run in minutes

    Args:
        tik (float): Time when total run started
        tik_tik (float): Time when the model for one random state started
        index (int): Which iteration index you are (starts at 0)

    Returns:
        Tuple: Tuple containing the timing information
    """

    tok = perf_counter()
    time_single = round((tok - tik_tik) / 60, 1)
    time_total = round((tok - tik) / 60, 1)
    time_avg = round(time_total / (index + 1), 1)
    return time_single, time_total, time_avg


def parallel_ranges(start: int, end: int, num_arrays: int) -> ndarray:
    """Splits range(start, end) in num_arrays different arrays.
    Example with num_arrays= 8
    ar1 = [1,9,17,etc], ar2 = [2,10,18,etc], ar3=[3,11,19,etc], etc.

    Args:
        start (int): Start of range
        end (int): End of range
        num_arrays (int): Number of different arrays produced

    Returns:
        ndarray: Array of num_arrays number of arrays
    """
    arrays = [[] for _ in range(num_arrays)]

    for i in range(start, end + 1):
        index = (i - start) % num_arrays
        arrays[index].append(i)
    return arrays


def combine_csvs(folder_path: Path):
    """Combines all json that are in the given folder into one

    Args:
        start (int): Start random state of combined
        end (int): End random state of combined
        folder_path (Path): where the json are and will be saved
    """
    # Create a list to hold all the JSON data
    combined_data = []

    # Loop through all files in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            # Convert each csv to a list of Estimated Parameters and those the the combined data list
            filepath = folder_path / filename
            data = csv_to_list(filepath)
            combined_data.extend(data)

    # Save the combined result back into a csv
    save_location_result = folder_path / f"combined_results.csv"
    save_to_csv(combined_data, save_location_result)
    print("Finished combining")


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
