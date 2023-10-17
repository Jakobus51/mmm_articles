from data_generation.custom_dgp import CustomDGP
from ssm_model.estimated_parameters import EstimatedParameters
from pandas import DataFrame
from scipy.optimize import curve_fit, minimize, Bounds
from numpy import average, inf, ndarray, ones
from ssm_model.constants import ModelSpecifications as ms
from data_generation.dg_constants import DataHelpers as dh
from helpers import apply_marketing_transformation_single


def get_optimal_budget(
    true_parameters: EstimatedParameters,
    marketing_data: DataFrame,
    date: str,
    budget_deviation: float,
):
    """Obtain the best budget allocation which maximizes your expected revenue,
    print the results

    Args:
        true_parameters (EstimatedParameters): True dgp
        marketing_data (DataFrame): Marketing data used for dgp
        date (str): Date you want to optimize
        budget_deviation (float): How much you can deviate from you original budget

    """
    day_to_be_optimized = marketing_data.loc[marketing_data["Date"] == date]
    original_budget = day_to_be_optimized[dh.FAKE_CHANNELS].values[0]

    # Constraint: You can not exceed the total budget
    def con(budget):
        return budget.sum() - original_budget.sum()

    res = minimize(
        fun=get_revenue_for_budget,
        x0=ones(ms.NUMBER_OF_BETAS),
        args=(true_parameters, marketing_data.copy()),
        bounds=setBounds(original_budget, budget_deviation),
        constraints=[{"type": "eq", "fun": con}],
        options={"disp": False},
    )
    new_budget = res.x
    original_revenue = -get_revenue_for_budget(
        original_budget, true_parameters, marketing_data
    )
    new_revenue = -get_revenue_for_budget(new_budget, true_parameters, marketing_data)
    return (original_revenue, new_revenue)
    # print_budget_results(true_parameters, marketing_data, original_budget, new_budget)


def get_revenue_for_budget(
    budget: ndarray, true_parameters: EstimatedParameters, marketing_data: DataFrame
) -> float:
    """Obtain the revenue based on a given budget

    Args:
        budget (ndarray): The budget of which you will calculate the ne revenue
        true_parameters (EstimatedParameters): Object which stores the saturation curves parameters
        marketing_data (DataFrame): Data of marketing expenditures

    Returns:
        float: the revenue for the budget * -1 since also used in minimization problem
    """
    total_revenue = 0
    for i, channel_spend in enumerate(budget):
        index = i + 1
        channel = f"Channel_{index}"
        phi = getattr(true_parameters, f"phi_{index}")
        rho = getattr(true_parameters, f"rho_{index}")
        theta = getattr(true_parameters, f"theta_{index}")
        beta = getattr(true_parameters, f"beta_{index}")

        response = apply_marketing_transformation_single(
            marketing_data[channel], channel_spend, theta, phi, rho
        )
        revenue = response * beta

        # Is a minimization problem so we negatively add the revenue per channel
        total_revenue -= revenue
    return total_revenue


def setBounds(original_budget: ndarray, deviation: float) -> list[tuple]:
    """Get the optimization bounds for each channel's budget

    Args:
        original_budget (ndarray): Original channel budget
        deviation (float): % how much we can differ from the original budget

    Returns:
        list[tuple]: the bounds
    """
    bounds = []
    for channel in original_budget:
        # SEM channel spend can't be changed @TODO for the real deal
        # if no deviation is given everything goes
        if deviation == 0:
            bound = (0, inf)
        # Otherwise it is capped
        else:
            bound = (channel / (1 + deviation), channel * (1 + deviation))
        bounds.append(bound)
    return list(bounds)


def print_budget_results(
    true_parameters: EstimatedParameters,
    marketing_data: DataFrame,
    original_budget: ndarray,
    new_budget: ndarray,
    model_number: str = None,
):
    original_revenue = -get_revenue_for_budget(
        original_budget, true_parameters, marketing_data
    )
    new_revenue = -get_revenue_for_budget(new_budget, true_parameters, marketing_data)
    print(f"\n====== Optimized Budget (model: {model_number}) ======\n")
    print(f"{'_':<10} {'Original':<10} {'New':<10} {'Difference':<10}")
    print(
        f"{'Revenue':<10} {original_revenue:<10.1f} {new_revenue:<10.1f} {get_perc_difference(original_revenue, new_revenue)}"
    )
    for index, (original, new) in enumerate(zip(original_budget, new_budget)):
        print(f"{f'Channel_{index}':<10} {original:<10.1f} {new:<10.1f}")


def get_perc_difference(old: float, new: float):
    perc = new / old * 100 - 100
    return f"+{perc:.2f}%" if perc >= 0 else f"{perc:.2f}%"


def get_optimal_budget_w_ssm(
    true_parameters: EstimatedParameters,
    ssm_parameters: EstimatedParameters,
    marketing_data: DataFrame,
    date: str,
    budget_deviation: float,
):
    """Obtain the best budget allocation which maximizes your expected revenue,
    print the results

    Args:
        true_parameters (EstimatedParameters): True dgp
        marketing_data (DataFrame): Marketing data used for dgp
        date (str): Date you want to optimize
        budget_deviation (float): How much you can deviate from you original budget

    """
    day_to_be_optimized = marketing_data.loc[marketing_data["Date"] == date]
    original_budget = day_to_be_optimized[dh.FAKE_CHANNELS].values[0]

    # Constraint: You can not exceed the total budget
    def con(budget):
        return budget.sum() - original_budget.sum()

    res = minimize(
        fun=get_revenue_for_budget,
        x0=ones(ms.NUMBER_OF_BETAS),
        args=(ssm_parameters, marketing_data.copy()),
        bounds=setBounds(original_budget, budget_deviation),
        constraints=[{"type": "eq", "fun": con}],
        options={"disp": False},
    )
    new_budget = res.x
    original_revenue = -get_revenue_for_budget(
        original_budget, true_parameters, marketing_data
    )
    new_revenue = -get_revenue_for_budget(new_budget, true_parameters, marketing_data)
    return (original_revenue, new_revenue)
    # print_budget_results(true_parameters, marketing_data, original_budget, new_budget)
