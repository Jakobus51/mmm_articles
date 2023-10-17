from data_generation.custom_dgp import CustomDGP
from pandas import read_csv, DataFrame
from ..constants import SaveLocation as sl
from budget_optimization import (
    get_optimal_budget_w_ssm,
)
from data_generation.fake_marketing import FakeMarketing
from time import perf_counter
from ssm_model.estimated_parameters import EstimatedParameters


def analyze_ssm_models(start_state: int, end_state: int):

    tik = perf_counter()
    random_states = range(start_state, end_state)


    budget_deviation = 0.25
    results_df = DataFrame(columns=["state", "original", "SSM", "SSM_increase"])

    # ======== PART 2, ANALYZE =============
    for random_state in random_states:
        tik = perf_counter()

        marketing_data = FakeMarketing(random_state).df

        dgp = CustomDGP(random_state, marketing_data, False, False)
        ssm_df = read_csv(sl.SSM_100RUNS)
        ssm_df_single = ssm_df[ssm_df["state"] == random_state]
        ssm_parameters = EstimatedParameters(False)
        ssm_parameters.set_via_array(ssm_df_single.values[0][1:])

        print(f"Started state {random_state}")

        # Get the original and optimal revenue due to marketing
        total_original_revenue = 0
        total_ssm_revenue = 0
        for date in dgp.dates:
            optimization_date = date.strftime("%Y-%m-%d")

            original, ssm_revenue = get_optimal_budget_w_ssm(
                dgp.true_parameters,
                ssm_parameters,
                marketing_data,
                optimization_date,
                budget_deviation,
            )
            total_original_revenue += original
            total_ssm_revenue += ssm_revenue

        new_row = [
            random_state,
            total_original_revenue,
            total_ssm_revenue,
            (total_ssm_revenue / total_original_revenue - 1),
        ]
        results_df.loc[len(results_df)] = new_row

        print(f"Finished state {random_state} in {int(perf_counter() - tik)}s")
    save_location = sl.ROBYN_100RUNS / "_budget_optimization_results_ssm.csv"
    results_df.to_csv(save_location)

    print(f"FINISHED BABY!!!!")
