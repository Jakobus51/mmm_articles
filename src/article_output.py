from matplotlib.pyplot import subplots, show, ioff
from data_generation.custom_dgp import CustomDGP
from pandas import DataFrame, read_csv, Series
from ssm_model.constants import ModelSpecifications as ms
from mpl_interactions import panhandler, zoom_factory
from data_generation.custom_dgp import CustomDGP
from pathlib import Path
from pandas import read_csv, concat
from ssm_model.constants import ModelSpecifications as ms
from numpy import array, sort, abs, row_stack, cumsum, nan
from helpers import (
    apply_marketing_transformation,
    combine_robyn_dfs,
    apply_saturation,
)
from constants import RobynModels
from collections import Counter
from data_generation.fake_marketing import FakeMarketing
from constants import SaveLocation as sl
import acmetric as ac


def compare_dgp_robyn_plot(dgp: CustomDGP, robyn_dfs: list[DataFrame]):
    """Plots of the signal
    0. LL (trend)
    1. yearly
    2. weekly
    3. Beta
    4. promo

    Args:
        ssm (StateSpaceModel): _description_
    """
    # First 4 plots
    with ioff():
        fig_1, axis = subplots(5, 1)
        fig_1.set_size_inches(12, 9)
        fig_1.set_tight_layout(True)
        # LL, trend
        axis[0].plot(
            dgp.dates,
            sum(dgp.alpha[:, i, 0] for i in range(0, 2)),
            label="True",
        )
        axis[0].axhline(0, color="k", linewidth=0.5)

        axis[0].set_title("Trend")

        # Yearly
        axis[1].plot(
            dgp.dates,
            sum(
                dgp.Z[:, 0, i] * dgp.alpha[:, i, 0]
                for i in range(
                    ms.INDEX_Y_GAMMA, ms.INDEX_Y_GAMMA + ms.NUMBER_OF_YEARLY_GAMMAS
                )
            ),
            label="True",
        )
        axis[1].axhline(0, color="k", linewidth=0.5)

        axis[1].set_title("Yearly seasonality")

        # Weekly
        axis[2].plot(
            dgp.dates,
            sum(
                dgp.Z[:, 0, i] * dgp.alpha[:, i, 0]
                for i in range(
                    ms.INDEX_W_GAMMA, ms.INDEX_W_GAMMA + ms.NUMBER_OF_WEEKLY_GAMMAS
                )
            ),
            label="True",
        )
        axis[2].axhline(0, color="k", linewidth=0.5)
        axis[2].set_title("Weekly seasonality")

        # Beta
        axis[3].plot(
            dgp.dates,
            sum(
                dgp.Z[:, 0, i] * dgp.alpha[:, i, 0]
                for i in range(ms.INDEX_BETA, ms.INDEX_BETA + ms.NUMBER_OF_BETAS)
            ),
            label="True",
        )

        axis[3].set_title("Marketing effects")

        # Promo
        axis[4].plot(
            dgp.dates,
            sum(
                dgp.Z[:, 0, i] * dgp.alpha[:, i, 0]
                for i in range(ms.INDEX_PROMO, ms.INDEX_PROMO + ms.NUMBER_OF_PROMOS)
            ),
            label="True",
        )
        axis[4].set_title("Promotional effects")

        for index, robyn_df in enumerate(robyn_dfs):
            model_number = robyn_df["MODEL_NUMBER"].iloc[0]
            axis[0].plot(dgp.dates, robyn_df["TREND"], label=model_number)
            axis[1].plot(dgp.dates, robyn_df["SEASON"], label=model_number)
            axis[2].plot(dgp.dates, robyn_df["WEEKDAY"], label=model_number)
            axis[3].plot(dgp.dates, robyn_df["MARKETING"], label=model_number)
            axis[4].plot(dgp.dates, robyn_df["HOLIDAY"], label=model_number)

        for ax in axis:
            ax.set_ylabel("Revenue")
            ax.legend(loc="upper left")

        # Enable zoom
        disconnect_zoom = zoom_factory(axis[0])
        disconnect_zoom = zoom_factory(axis[1])
        disconnect_zoom = zoom_factory(axis[2])
        disconnect_zoom = zoom_factory(axis[3])
        disconnect_zoom = zoom_factory(axis[4])

    # Enable pan
    pan_handler = panhandler(fig_1)
    show(block=False)


def print_prediction_stacked_graph(dgp: CustomDGP):
    """Graph where the total revenue is split up per different element

    Args:
        dgp (CustomDGP): _description_
    """
    local_level = sum(dgp.Z[:, 0, i] * dgp.alpha[:, i, 0] for i in range(0, 2))
    ask = sum(
        dgp.Z[:, 0, i] * dgp.alpha[:, i, 0]
        for i in range(ms.INDEX_Y_GAMMA, ms.INDEX_Y_GAMMA + ms.NUMBER_OF_YEARLY_GAMMAS)
    )
    weekly = sum(
        dgp.Z[:, 0, i] * dgp.alpha[:, i, 0]
        for i in range(ms.INDEX_W_GAMMA, ms.INDEX_W_GAMMA + ms.NUMBER_OF_WEEKLY_GAMMAS)
    )
    beta = sum(
        dgp.Z[:, 0, i] * dgp.alpha[:, i, 0]
        for i in range(ms.INDEX_BETA, ms.INDEX_BETA + ms.NUMBER_OF_BETAS)
    )
    promo = sum(
        dgp.Z[:, 0, i] * dgp.alpha[:, i, 0]
        for i in range(ms.INDEX_PROMO, ms.INDEX_PROMO + ms.NUMBER_OF_PROMOS)
    )

    y = row_stack((local_level, ask, weekly, beta, promo))
    y_stack = cumsum(y, axis=0)
    x = dgp.dates

    with ioff():
        fig_1, axis = subplots(1, 1)
        fig_1.set_size_inches(11, 9)
        fig_1.set_tight_layout(True)

        axis.fill_between(x, y_stack[0, :], y2=0, alpha=0.7)
        axis.fill_between(x, y_stack[0, :], y_stack[1, :], alpha=0.7)
        axis.fill_between(x, y_stack[1, :], y_stack[2, :], alpha=0.7)
        axis.fill_between(x, y_stack[2, :], y_stack[3, :], alpha=0.7)
        axis.fill_between(x, y_stack[3, :], y_stack[4, :], alpha=0.7)
        axis.legend(["Trend", "Yearly", "Weekly", "Marketing", "Promo"])

        disconnect_zoom = zoom_factory(axis)

    # Enable pan
    pan_handler = panhandler(fig_1)
    show(block=False)


def print_saturation_curves(random_state: int):
    marketing_data = FakeMarketing(random_state).df
    dgp = CustomDGP(random_state, marketing_data, False, False)

    # Get the robyn data
    robyn_hyperparam_df = read_csv(
        sl.ROBYN_100RUNS / f"hyperparam_(state={random_state}).csv"
    )
    robyn_beta_df = read_csv(sl.ROBYN_100RUNS / f"beta_(state={random_state}).csv")
    dfs = []

    # Loop over the different robyn models and creater the dfs that contain all the hyperparamters needed for the saturation curves
    for model_number in robyn_hyperparam_df["solID"]:
        combined_robyn_df = combine_robyn_dfs(
            robyn_hyperparam_df, robyn_beta_df, model_number
        )
        dfs.append(combined_robyn_df)

    # Concatenate the list of dataframes into one large dataframe
    robyn_df = concat(dfs, ignore_index=True)

    ssm_df = read_csv(sl.SSM_100RUNS)
    ssm_df_single = ssm_df[ssm_df["state"] == random_state]

    fig_1, axis = subplots(4, 2)
    fig_1.set_size_inches(12, 9)
    fig_1.set_tight_layout(True)

    # For each channel plot the saturatio curve
    for i in range(1, 9):
        _plot_channel_response(
            i,
            axis[(i - 1) % 4][(i - 1) // 4],
            dgp,
            marketing_data,
            robyn_df,
            ssm_df_single,
        )
    handles, labels = axis[0][0].get_legend_handles_labels()
    fig_1.legend(handles, labels, loc="upper right")

    show(block=True)


def _plot_channel_response(
    i: int,
    ax,
    dgp: CustomDGP,
    marketing_data: DataFrame,
    robyn_df: DataFrame,
    ssm_df: DataFrame,
):
    channel = f"Channel_{i}"

    # True parameter graphs
    x = array(range(0, int(marketing_data[channel].max())))
    phi_true = getattr(dgp.true_parameters, f"phi_{i}")
    rho_true = getattr(dgp.true_parameters, f"rho_{i}")
    theta_true = getattr(dgp.true_parameters, f"theta_{i}")
    beta_true = getattr(dgp.true_parameters, f"beta_{i}")
    response = apply_saturation(x, phi_true, rho_true * (marketing_data[channel].max()))
    ax.plot(x, response * beta_true, label="True", color=ac.colors.sky, linewidth=2)

    color_map = {
        0: ac.colors.sun,
        1: ac.colors.coral,
        2: ac.colors.coral_60,
        3: ac.colors.sun_60,
    }

    # # SSM
    # phi_ssm = ssm_df[f"phi_{i}"].values[0]
    # rho_ssm = ssm_df[f"rho_{i}"].values[0]
    # theta_ssm = ssm_df[f"theta_{i}"].values[0]
    # beta_ssm = ssm_df[f"beta_{i}"].values[0]
    # response = apply_saturation(x, phi_ssm, rho_ssm * (marketing_data[channel].max()))
    # ax.plot(x, response * beta_ssm, label="SSM")

    # Robyn models
    for index, row in robyn_df[:4].iterrows():
        beta = row[f"Channel_{i}"]
        phi = row[f"Channel_{i}_alphas"]
        rho = row[f"Channel_{i}_gammas"]
        theta = row[f"Channel_{i}_thetas"]
        response = apply_saturation(x, phi, rho * (marketing_data[channel].max()))
        ax.plot(
            x,
            response * beta,
            label=f"Robyn_{index}",
            color=color_map[index],
            linewidth=2,
        )

    ax.set_ylabel("Response")
    ax.set_xlabel("Spend")
    ax.set_title(channel)
    # ax.legend(loc="upper left")


def print_combined_saturation_curve(random_state: int):
    df = get_combined_curves(random_state)
    spend_to_check = 1200
    print(df[spend_to_check : spend_to_check + 1].iloc[:, 5:])

    fig_1, axis = subplots()
    fig_1.set_size_inches(8, 5)
    fig_1.set_tight_layout(True)
    axis.plot(df["Spend"], df["True"].values, label="True")
    axis.plot(df["Spend"], df["SSM"].values, label="SSM")
    axis.plot(df["Spend"], df["Robyn_0"].values, label="Robyn_0")
    axis.plot(df["Spend"], df["Robyn_1"].values, label="Robyn_1")
    axis.plot(df["Spend"], df["Robyn_2"].values, label="Robyn_2")
    axis.plot(df["Spend"], df["Robyn_3"].values, label="Robyn_3")
    axis.axvline(spend_to_check, color="red", alpha=0.3)
    axis.legend()

    axis.set_ylabel("Response")
    axis.set_xlabel("Spend")
    axis.set_title("Saturation curve for all marketing channels combined")
    show(block=True)


def get_combined_curves(random_state: int) -> DataFrame:
    marketing_data = FakeMarketing(random_state).df
    dgp = CustomDGP(random_state, marketing_data, False, False)

    robyn_hyperparam_df = read_csv(
        sl.ROBYN_100RUNS / f"hyperparam_(state={random_state}).csv"
    )
    robyn_beta_df = read_csv(sl.ROBYN_100RUNS / f"beta_(state={random_state}).csv")
    dfs = []
    for model_number in robyn_hyperparam_df["solID"]:
        combined_robyn_df = combine_robyn_dfs(
            robyn_hyperparam_df, robyn_beta_df, model_number
        )
        dfs.append(combined_robyn_df)

    # Concatenate the list of dataframes into one large dataframe
    robyn_df = concat(dfs, ignore_index=True)

    ssm_df = read_csv(sl.SSM_100RUNS)
    ssm_df_single = ssm_df[ssm_df["state"] == random_state]

    max_spend = 1400
    x = array(range(0, max_spend))
    data = {
        "Spend": [0] * max_spend,
        "True": [0] * max_spend,
        "SSM": [0] * max_spend,
        "Robyn_0": [0] * max_spend,
        "Robyn_1": [0] * max_spend,
        "Robyn_2": [0] * max_spend,
        "Robyn_3": [0] * max_spend,
    }
    df = DataFrame(data)

    df["Spend"] = x

    for i in range(1, 9):
        channel = f"Channel_{i}"

        # True
        phi_true = getattr(dgp.true_parameters, f"phi_{i}")
        rho_true = getattr(dgp.true_parameters, f"rho_{i}")
        theta_true = getattr(dgp.true_parameters, f"theta_{i}")
        beta_true = getattr(dgp.true_parameters, f"beta_{i}")
        response = apply_saturation(
            x, phi_true, rho_true * (marketing_data[channel].max())
        )
        df["True"] += Series(response * beta_true)

        # SSM
        phi_ssm = ssm_df[f"phi_{i}"].values[0]
        rho_ssm = ssm_df[f"rho_{i}"].values[0]
        theta_ssm = ssm_df[f"theta_{i}"].values[0]
        beta_ssm = ssm_df[f"beta_{i}"].values[0]
        response = apply_saturation(
            x, phi_ssm, rho_ssm * (marketing_data[channel].max())
        )
        df["SSM"] += Series(response * beta_ssm)

        # Robyn models
        for index, row in robyn_df[:4].iterrows():
            beta = row[f"Channel_{i}"]
            phi = row[f"Channel_{i}_alphas"]
            rho = row[f"Channel_{i}_gammas"]
            theta = row[f"Channel_{i}_thetas"]
            response = apply_saturation(x, phi, rho * (marketing_data[channel].max()))
            df[f"Robyn_{index}"] += Series(response * beta)

    df["True_marginal"] = df["True"].shift(-1) - df["True"]
    df["SSM_marginal"] = df["SSM"].shift(-1) - df["SSM"]
    df["Robyn_0_marginal"] = df["Robyn_0"].shift(-1) - df["Robyn_0"]
    df["Robyn_1_marginal"] = df["Robyn_1"].shift(-1) - df["Robyn_1"]
    df["Robyn_2_marginal"] = df["Robyn_2"].shift(-1) - df["Robyn_2"]
    df["Robyn_3_marginal"] = df["Robyn_3"].shift(-1) - df["Robyn_3"]

    return df


def analyze_budget_optimization(result_location: Path):
    """Check how often and how many robyn models lead to a decrease in revenue

    Args:
        result_location (Path): Location where the optimized budgets per model are
    """
    results = read_csv(result_location)

    # Select all the percentual columns since theses are the ones we want to chekc
    column_names = []
    for model in RobynModels.MODELS:
        column_name = f"{model}_perc"
        column_names.append(column_name)

    negative_counts = []
    for index, row in results.iterrows():
        # Count the negative values in the specified columns
        negative_count = sum(1 for col in column_names if "-" in str(row[col]))
        negative_counts.append(negative_count)

    # Count how often models cause negative revenue per day
    counter = Counter(negative_counts)

    print("Occurrences of each entry in the negative_counts list:")
    for entry, count in counter.items():
        print(f"{entry}: {count} times")


def budget_optimization_result():
    df_robyn = read_csv(
        "/Users/jakob/Documents/ACMetric/MMM_article/100_runs/_results_100_runs.csv"
    )
    df_ssm = read_csv(
        "/Users/jakob/Documents/ACMetric/MMM_article/100_runs/_budget_optimization_results_ssm.csv"
    )

    print(df_robyn["optimal increase"].max())
    print(df_ssm["SSM_increase"].max())

    # Multiply everything by 100 since we show percentages
    optimal_mean = df_robyn["optimal increase"].mean() * 100
    optimal_std = df_robyn["optimal increase"].std() * 100
    robyn_mean = df_robyn["robyn increase"].mean() * 100
    robyn_std = df_robyn["robyn increase"].std() * 100
    ssm_mean = df_ssm["SSM_increase"].mean() * 100
    ssm_std = df_ssm["SSM_increase"].std() * 100

    ten_perc = int(len(df_robyn) * 0.10)
    top_10_nrmse_rows = df_robyn.nsmallest(ten_perc, "nrmse_test")
    top_10_rsq_rows = df_robyn.nlargest(ten_perc, "rsq_test")
    top_10_decomp_rows = df_robyn.nsmallest(ten_perc, "decomp.rssd")

    top_10_nrmse_mean = top_10_nrmse_rows["robyn increase"].mean() * 100
    top_10_nrmse_std = top_10_nrmse_rows["robyn increase"].std() * 100
    top_10_rsq_mean = top_10_rsq_rows["robyn increase"].mean() * 100
    top_10_rsq_std = top_10_rsq_rows["robyn increase"].std() * 100
    top_10_decomp_mean = top_10_decomp_rows["robyn increase"].mean() * 100
    top_10_decomp_std = top_10_decomp_rows["robyn increase"].std() * 100

    labels = [
        "Optimal",
        "Robyn",
        "Top 10%\nNRMSE",
        "Top 10%\nR2",
        "Top 10%\nDRSSD",
        # "SSM",
    ]

    values = [
        optimal_mean,
        robyn_mean,
        top_10_nrmse_mean,
        top_10_rsq_mean,
        top_10_decomp_mean,
        # ssm_mean,
    ]
    errors = [
        optimal_std,
        robyn_std,
        top_10_nrmse_std,
        top_10_rsq_std,
        top_10_decomp_std,
        # ssm_std,
    ]

    # Plot
    fig, ax = subplots(figsize=(10, 6))

    bar_colors = [
        ac.colors.sky,
        ac.colors.coral,
        ac.colors.sun,
        ac.colors.sun,
        ac.colors.sun,
    ]

    bars = ax.bar(
        labels,
        values,
        yerr=errors,
        color=bar_colors,
        align="center",
        ecolor=ac.colors.stone,
        capsize=10,
    )

    # Set the y-axis ticks to reflect percentages
    ax.set_ylabel("Increase (%)")
    # ax.yaxis.grid(True)
    # plt.xticks(rotation = -45) # Rotates X-Axis Ticks by 45-degrees
    ax.set_title("Average revenue due to marketing increase  (±1 std)")
    # ax.legend(["Mean increase"], loc="upper center")
    # ax.set_ylim(top=12)
    ax.axhline(0, color="k", linewidth=0.5)

    show()


from turtle import color
from helpers import (
    apply_marketing_transformation,
    apply_marketing_transformation_single,
)
from matplotlib.pyplot import subplots, show, ioff


def plot_it():
    x = range(0, 1001)
    theta = 0.2
    fig_1, axis = subplots(1, 1)
    fig_1.set_size_inches(9, 6)
    fig_1.set_tight_layout(True)
    y_0 = apply_marketing_transformation(x, theta, phi=1.25, rho=0.3)
    y_1 = apply_marketing_transformation(x, theta, phi=2.5, rho=0.3)
    y_2 = apply_marketing_transformation(x, theta, phi=1.25, rho=0.7)
    y_3 = apply_marketing_transformation(x, theta, phi=2.5, rho=0.7)

    # Observation and predictions
    axis.plot(x, y_0, label=f"phi={1.25}, rho={0.3}")
    axis.plot(x, y_1, label=f"phi={2.5}, rho={0.3}")
    axis.plot(x, y_2, label=f"phi={1.25}, rho={0.7}")
    axis.plot(x, y_3, label=f"phi={2.5}, rho={0.7}")
    axis.legend()
    axis.set_xlabel("Marketing Spend")
    axis.set_ylabel("Response")

    # PLOTTING RHO, PHI PAIRS
    #     plt.figure()
    # plt.scatter(rho, phi)
    # for x,y, label in zip (rho, phi, range(1, len(phi)+1)):
    #     plt.text(x,y,label)


def plot_budget_optimization(channel_1_spend, channel_2_spend):
    x = range(0, 1001)
    theta = 0.0001
    fig_1, axis = subplots()
    fig_1.set_size_inches(9, 6)
    fig_1.set_tight_layout(True)
    y_0 = apply_marketing_transformation(x, theta, phi=1.25, rho=0.3) * 1000
    y_1 = apply_marketing_transformation(x, theta, phi=2.5, rho=0.7) * 1000

    color_1 = ac.colors.coral
    color_2 = ac.colors.sun
    axis.plot(
        x,
        y_0,
        label=f"Channel 1",
        color=color_1,
    )
    axis.plot(
        x,
        y_1,
        label=f"Channel 2",
        color=color_2,
    )

    # Original points
    original_x_0 = channel_1_spend
    original_y_0 = (
        apply_marketing_transformation_single(x, original_x_0, theta, phi=1.25, rho=0.3)
        * 1000
    )
    plot_point_lines(axis, original_x_0, original_y_0, color_1)

    original_x_1 = channel_2_spend
    original_y_1 = (
        apply_marketing_transformation_single(x, original_x_1, theta, phi=2.5, rho=0.7)
        * 1000
    )
    plot_point_lines(axis, original_x_1, original_y_1, color_2)

    axis.set_ylim(bottom=0)
    axis.set_xlim(left=0)

    axis.legend()
    axis.set_xlabel("Marketing Spend")
    axis.set_ylabel("Response")

    # PLOTTING RHO, PHI PAIRS
    #     plt.figure()
    # plt.scatter(rho, phi)
    # for x,y, label in zip (rho, phi, range(1, len(phi)+1)):
    #     plt.text(x,y,label)


def plot_point_lines(axis, original_x, original_y, color):
    # Original budget

    axis.vlines(
        x=original_x,
        ymin=0,
        ymax=original_y,
        color=color,
        ls=":",
    )

    axis.hlines(
        y=original_y,
        xmin=0,
        xmax=original_x,
        color=color,
        ls=":",
    )
    offset = 7
    axis.annotate(
        str(int(original_y)), xy=(0 + offset, original_y + offset), color=color
    )
    axis.annotate(
        str(int(original_x)), xy=(original_x + offset, 0 + offset), color=color
    )


if __name__ == "__main__":
    random_state = 67
    print_saturation_curves(random_state)
    # print_combined_saturation_curve(random_state)
    # budget_optimization_result()
    # marketing_data = FakeMarketing(random_state).df
    # dgp = CustomDGP(random_state, marketing_data, False, True)
    # plot_budget_optimization(340, 660)
    # plot_budget_optimization(100, 900)
    show()
