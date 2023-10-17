from data_generation.custom_dgp import CustomDGP
from helpers import (
    combine_robyn_dfs
)
from pandas import read_csv, DataFrame
from constants import SaveLocation as sl
from budget_optimization import get_optimal_budget
from budget_optimization_w_robyn import run_robyn_new_budget
from data_generation.fake_marketing import FakeMarketing
from time import perf_counter


def analyze_robyn_models(start_state: int, end_state: int):
    tik = perf_counter()
    random_states = range(start_state, end_state)


    budget_deviation = 0.25
    results_df = DataFrame(
        columns=[
            "state",
            "model_number",
            "original",
            "optimal",
            "robyn",
            "optimal increase",
            "robyn increase",
        ]
    )

    # ======== PART 2, ANALYZE =============
    for random_state in random_states:
        tik = perf_counter()

        marketing_data = FakeMarketing(random_state).df

        dgp = CustomDGP(random_state, marketing_data, True, False)
        robyn_hyperparam_df = read_csv(
            sl.ROBYN_100RUNS / f"hyperparam_(state={random_state}).csv"
        )
        robyn_beta_df = read_csv(sl.ROBYN_100RUNS / f"beta_(state={random_state}).csv")

        print(
            f"Started state {random_state} which contains {len(robyn_hyperparam_df)} models"
        )

        # Get the original and optimal revenue due to marketing
        total_original_revenue = 0
        total_optimal_revenue = 0
        for date in dgp.dates:
            optimization_date = date.strftime("%Y-%m-%d")

            original, optimal = get_optimal_budget(
                dgp.true_parameters, marketing_data, optimization_date, budget_deviation
            )
            total_original_revenue += original
            total_optimal_revenue += optimal

        print(f"Finished optimal for state {random_state} in {int(perf_counter() - tik)}s")
        # Do the same but now for the top solutions of the robyn run for the given random state
        for model_number in robyn_hyperparam_df["solID"]:
            tiktik = perf_counter()
            total_robyn_optimal = 0
            combined_robyn_df = combine_robyn_dfs(
                robyn_hyperparam_df, robyn_beta_df, model_number
            )

            # Get the optimal for the given robyn model
            for date in dgp.dates:
                optimization_date = date.strftime("%Y-%m-%d")

                model_optimal = run_robyn_new_budget(
                    dgp.true_parameters,
                    combined_robyn_df,
                    model_number,
                    marketing_data,
                    optimization_date,
                    budget_deviation,
                )
                total_robyn_optimal += model_optimal

            new_row = [
                random_state,
                model_number,
                total_original_revenue,
                total_optimal_revenue,
                total_robyn_optimal,
                (total_optimal_revenue / total_original_revenue - 1),
                (total_robyn_optimal / total_original_revenue - 1),
            ]
            results_df.loc[len(results_df)] = new_row

            print(
                f"Finished robyn model {model_number} for state {random_state} in {int(perf_counter() - tiktik)}s"
            )
        print(f"Finished state {random_state} in {int(perf_counter() - tik)}s")
        save_location = sl.ROBYN_100RUNS / "_budget_optimization_results.csv"
        results_df.to_csv(save_location)

    print(f"FINISHED BABY!!!!")
