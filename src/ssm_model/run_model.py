from pathlib import Path
from ssm_model.estimated_parameters import EstimatedParameters
from ssm_model.estimated_parameters_bounds import EstimatedParametersBounds
from ssm_model.state_space_model import StateSpaceModel
from data_generation.custom_dgp import CustomDGP
from data_generation.fake_marketing import FakeMarketing
from ssm_model.multicore_helpers import (
    timing_run,
    parallel_ranges,
    save_to_csv,
    combine_csvs,
)
from pandas import read_csv, DataFrame
from multiprocessing import Pool
from functools import partial
from time import perf_counter
import logging
from numpy import ndarray
import os


def run_model(
    state_range: ndarray,
    starting_parameters: EstimatedParameters,
    bounds: EstimatedParametersBounds,
):
    """Runs the model for the the different random states in in the state range

    Args:
        state_range (ndarray): Array with random states for which the model will be used
        starting_parameters (EstimatedParameters): The parameters used as initial optimization values (stay fixed for all random states)
        bounds (EstimatedParametersBounds): The bounds used as initial optimization values (stay fixed for all random states)
        marketing_data (DataFrame): The fake marketing data used in the model (stay fixed for all random states)
    """
    # Set logger (needs to be set in the function since multiprocessing)
    logging.basicConfig(
        level=logging.DEBUG,
        format="(%(asctime)s) %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )

    # List where the results will be appended to
    estimated_parameters_list = []

    # Each process writes to its own file to mitigate race condition error
    save_location = f"/Users/jakob/Documents/ACMetric/MMM_article/100_runs/ssm_100_runs/parameters_chunk_{state_range[0]}_{state_range[-1]}.csv"

    tik = perf_counter()
    logging.info(f"Started simulation for chunk: {state_range[0]}_{state_range[-1]}")

    # Run the model for each random state and save the results
    for index, random_state in enumerate(state_range):
        tik_tik = perf_counter()

        marketing_data = FakeMarketing(random_state).df

        dgp = CustomDGP(random_state, marketing_data, False, False)
        model = StateSpaceModel(dgp.y)

        model.train_model(
            starting_parameters.get_values(),
            bounds.get_bounds(),
            marketing_data,
            400,
        )

        # Save the estimated parameters to the list
        estimated_parameters_list.append(model.parameters.estimated_parameters)

        # Save to csv after each run so you keep your results if something goes wrong midway
        save_to_csv(estimated_parameters_list, save_location)

        # Some logging info about the run
        run_single, run_total, run_avg = timing_run(tik, tik_tik, index)
        logging.info(
            f"Finished state {random_state}, run {index + 1} (time: {run_single} min, total: {run_total} min, avg: {run_avg} min)"
        )

def run_ssm_model_for_range(start_range: int, end_range: int):
    """Run the ssm model for different random states in a given range

    Args:
        start_range (int): _description_
        end_range (int): _description_
    """    
    # Use partial to fix the parameters that stay the same for each function call
    run_model_fixed = partial(
        run_model,
        starting_parameters=EstimatedParameters(True),
        bounds=EstimatedParametersBounds(),
    )

    # Set the number of cores that will be used
    number_of_cores = os.cpu_count()
    pool = Pool(number_of_cores)
   
    # Divide the random state range over the available cores
    state_ranges = parallel_ranges(start_range, end_range-1, number_of_cores)

    # Map the divided random states to workers (=different cores)
    pool.map(run_model_fixed, state_ranges)

    # clean up (close prevents pool from excepting new tasks, join waits for all the tasks to have finished)
    pool.close()
    pool.join()

    combine_csvs(
        Path("/Users/jakob/Documents/ACMetric/MMM_article/100_runs/ssm_100_runs")
    )
    logging.info(f"Finished!")
