from data_generation.custom_dgp import CustomDGP
from ssm_model.estimated_parameters import EstimatedParameters
from pandas import DataFrame
from scipy.optimize import curve_fit, minimize, Bounds
from numpy import average, inf, ndarray, ones
from ssm_model.constants import ModelSpecifications as ms
from data_generation.dg_constants import DataHelpers as dh
from helpers import apply_marketing_transformation_single
from budget_optimization import setBounds, print_budget_results, get_revenue_for_budget


def run_robyn_new_budget(
    true_parameters: EstimatedParameters,
    robyn_param_df: DataFrame,
    model_number: str,
    marketing_data: DataFrame,
    optimization_date: str,
    budget_deviation: float,
):
    """Finds the revenue for a given day for the robyn model

    Args:
        true_parameters (EstimatedParameters): _description_
        robyn_param_df (DataFrame): _description_
        model_number (str): _description_
        marketing_data (DataFrame): _description_
        optimization_date (str): _description_
        budget_deviation (float): _description_

    Returns:
        _type_: _description_
    """
    new_budget = get_optimal_budget_w_robyn(
        robyn_param_df.loc[robyn_param_df["solID"] == model_number],
        marketing_data,
        optimization_date,
        budget_deviation,
    )
    new_revenue = -get_revenue_for_budget(new_budget, true_parameters, marketing_data)
    return new_revenue

    # print_budget_results(
    #     true_parameters, marketing_data, original_budget, new_budget, model_number
    # )


def get_optimal_budget_w_robyn(
    robyn_df: DataFrame,
    marketing_data: DataFrame,
    date: str,
    budget_deviation: float,
) -> ndarray:
    """Obtain the best budget allocation which maximizes your expected revenue,
    print the results

    Args:
        robyn_df (DataFrame): True dgp
        marketing_data (DataFrame): Marketing data used for dgp
        date (str): Date you want to optimize
        budget_deviation (float): How much you can deviate from you original budget

    Returns:
        ndarray: the optimized budget
    """
    day_to_be_optimized = marketing_data.loc[marketing_data["Date"] == date]
    original_budget = day_to_be_optimized[dh.FAKE_CHANNELS].values[0]

    # Constraint: You can not exceed the total budget
    def con(budget):
        return budget.sum() - original_budget.sum()

    res = minimize(
        fun=get_revenue_for_budget_w_robyn,
        x0=ones(ms.NUMBER_OF_BETAS),
        args=(robyn_df, marketing_data.copy()),
        bounds=setBounds(original_budget, budget_deviation),
        constraints=[{"type": "eq", "fun": con}],
        options={"disp": False},
    )
    return res.x


def get_revenue_for_budget_w_robyn(
    budget: ndarray, robyn_df: DataFrame, marketing_data: DataFrame
) -> float:
    """Obtain the revenue based on a given budget

    Args:
        budget (ndarray): The budget of which you will calculate the ne revenue
        robyn_df (DataFrame): Object which stores the saturation curves parameters
        marketing_data (DataFrame): Data of marketing expenditures

    Returns:
        float: the revenue for the budget * -1 since also used in minimization problem
    """
    total_revenue = 0
    for i, channel_spend in enumerate(budget):
        index = i + 1
        channel = f"Channel_{index}"
        beta = robyn_df[channel].values[0]
        phi = robyn_df[f"Channel_{index}_alphas"].values[0]
        rho = robyn_df[f"Channel_{index}_gammas"].values[0]
        theta = robyn_df[f"Channel_{index}_thetas"].values[0]

        response = apply_marketing_transformation_single(
            marketing_data[channel], channel_spend, theta, phi, rho
        )
        revenue = response * beta

        # Is a minimization problem so we negatively add the revenue per channel
        total_revenue -= revenue
    return total_revenue
