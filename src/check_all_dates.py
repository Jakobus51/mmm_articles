from data_generation.custom_dgp import CustomDGP
from helpers import (
    save_to_csv,
    save_dataset,
    get_robyn_parameters,
    get_robyn_predictions,
    make_column_names,
)
from pandas import read_csv, DataFrame
from data_generation.dg_constants import DataLocations as dl
from constants import SaveLocation as sl
from ssm_model.state_space_model import StateSpaceModel
from ssm_model.estimated_parameters import EstimatedParameters
from ssm_model.estimated_parameters_bounds import EstimatedParametersBounds
from ssm_model.constants import Optimization as op
from article_output import (
    print_prediction_stacked_graph,
    print_saturation_curves,
    compare_dgp_robyn_plot,
)
from matplotlib.pyplot import subplots, show, ioff
from pathlib import Path
from constants import RobynModels
from budget_optimization import get_optimal_budget, get_perc_difference
from budget_optimization_w_robyn import run_robyn_new_budget

random_state = 64
budget_deviation = 0.2
marketing_data = read_csv(dl.FAKE_MARKETING)

# Generate the dgp with the given random state and read-in robyn stuff
dgp = CustomDGP(random_state, marketing_data, True, False)
robyn_param_df = get_robyn_parameters(RobynModels.MODELS)


results_df = DataFrame(columns=make_column_names(RobynModels.MODELS))

# Loop over each date and get the precicted revenue per model
for date in dgp.dates:
    optimization_date = date.strftime("%Y-%m-%d")
    new_row = []
    new_row.append(date)

    original, optimal = get_optimal_budget(
        dgp.true_parameters, marketing_data, optimization_date, budget_deviation
    )
    new_row.append(original)
    new_row.append(optimal)
    new_row.append(get_perc_difference(original, optimal))

    for model_number in RobynModels.MODELS:
        model_optimal = run_robyn_new_budget(
            dgp.true_parameters,
            robyn_param_df,
            model_number,
            marketing_data,
            optimization_date,
            budget_deviation,
        )
        new_row.append(model_optimal)
        new_row.append(get_perc_difference(original, model_optimal))

    results_df.loc[len(results_df)] = new_row
    print(f"Finished {optimization_date}")

save_loaciton = sl.ROBYN_MODELS / "budget_optimization_results.csv"
results_df.to_csv(save_loaciton)
