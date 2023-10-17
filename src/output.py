from matplotlib.pyplot import subplots, show, ioff
from ssm_model.state_space_model import StateSpaceModel
from ssm_model.scores import LL_optimization_function
from numpy import array, sort, abs, row_stack, cumsum
from ssm_model.constants import DiagnosticTests
from seaborn import histplot
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_pacf
from data_generation.custom_dgp import CustomDGP
from pandas import DataFrame, read_csv
from data_generation.dg_constants import DataLocations as dl
from ssm_model.constants import ModelSpecifications as ms
from mpl_interactions import panhandler, zoom_factory
from pandas import concat, Series


class outputModel:
    dgp: CustomDGP  # Object containing information about the Data Generation Process
    ssm: StateSpaceModel  # Object containing all the LL optimized parameters

    def __init__(
        self,
        ssm,
        estimated_parameters,
        prediction_breakdown_graphs,
        prediction_breakdown_stacked_graph,
        scores,
        prediction_graphs,
        forecaster_graphs,
        diagnostic_values,
        diagnostic_graphs,
        check_errors,
    ):
        if estimated_parameters:
            self._print_parameters(ssm)
        if prediction_breakdown_graphs:
            self._prediction_breakdown_graphs(ssm)
        if prediction_breakdown_stacked_graph:
            self._print_prediction_stacked_graph(ssm)
        if scores:
            self._print_scores(ssm)
        if prediction_graphs:
            self._print_prediction_graphs(ssm)
        if forecaster_graphs:
            self._print_forecast_graphs(ssm)
        if diagnostic_values:
            self._print_diagnostic_values(ssm)
        if diagnostic_graphs:
            self._print_diagnostic_graphs(ssm)
        if check_errors:
            self._check_errors(ssm)

    def _print_parameters(self, ssm: StateSpaceModel):
        """Print the values of the estimated parameters

        Args:
            ssm (StateSpaceModel): Object containing all the LL optimized parameters
        """

        # Retrieve all the variables from the class and use those to display their values
        estimated_parameters = vars(ssm.parameters.estimated_parameters)

        for key in estimated_parameters:
            print(f"{key:<20}{estimated_parameters[key]:.3f}")

    def _prediction_breakdown_graphs(self, ssm: StateSpaceModel):
        """Plots of the signal
        0. Observations
        1. LL
        2. ASK
        3. Weekly
        4. Beta's
        5. promo


        Args:
            ssm (StateSpaceModel): _description_
        """
        # First 4 plots
        with ioff():
            fig_1, axis = subplots(4, 1)
            fig_1.set_size_inches(12, 9)
            fig_1.set_tight_layout(True)
            # Observation and predictions
            axis[0].plot(ssm.parameters.dates, ssm.parameters.y)
            axis[0].plot(
                ssm.parameters.dates,
                (ssm.parameters.Z @ ssm.filter.predicted_alpha)[:, 0, 0],
                "r--",
            )
            axis[0].set_title("Observations and predictions")

            # LL
            axis[1].plot(
                ssm.parameters.dates,
                sum(
                    ssm.parameters.Z[:, 0, i] * ssm.filter.predicted_alpha[:, i, 0]
                    for i in range(0, 2)
                ),
            )
            axis[1].axhline(0, color="k", linewidth=0.5)
            axis[1].set_title("Local Linear Level")

            # ASK
            axis[2].plot(
                ssm.parameters.dates,
                sum(
                    ssm.parameters.Z[:, 0, i] * ssm.filter.predicted_alpha[:, i, 0]
                    for i in range(ms.INDEX_ASK, ms.INDEX_ASK + ms.NUMBER_OF_ASK)
                ),
            )
            axis[2].axhline(0, color="k", linewidth=0.5)
            axis[2].set_title("ASK")

            # Weekly
            axis[3].plot(
                ssm.parameters.dates,
                sum(
                    ssm.parameters.Z[:, 0, i] * ssm.filter.predicted_alpha[:, i, 0]
                    for i in range(
                        ms.INDEX_W_GAMMA, ms.INDEX_W_GAMMA + ms.NUMBER_OF_WEEKLY_GAMMAS
                    )
                ),
            )
            axis[3].axhline(0, color="k", linewidth=0.5)
            axis[3].set_title("Weekly")

            # Enable zoom
            disconnect_zoom = zoom_factory(axis[0])
            disconnect_zoom = zoom_factory(axis[1])
            disconnect_zoom = zoom_factory(axis[2])
            disconnect_zoom = zoom_factory(axis[3])

        # Enable pan
        pan_handler = panhandler(fig_1)
        show(block=False)

        # Second figure with other 3 plots
        with ioff():
            fig_2, axis = subplots(2, 1)
            fig_2.set_size_inches(12, 9)
            fig_2.set_tight_layout(True)

            # Beta
            axis[0].plot(
                ssm.parameters.dates,
                sum(
                    ssm.parameters.Z[:, 0, i] * ssm.filter.predicted_alpha[:, i, 0]
                    for i in range(ms.INDEX_BETA, ms.INDEX_BETA + ms.NUMBER_OF_BETAS)
                ),
            )
            axis[0].axhline(0, color="k", linewidth=0.5)
            axis[0].set_title("Beta's")

            # promo
            axis[1].plot(
                ssm.parameters.dates,
                sum(
                    ssm.parameters.Z[:, 0, i] * ssm.filter.predicted_alpha[:, i, 0]
                    for i in range(ms.INDEX_PROMO, ms.INDEX_PROMO + ms.NUMBER_OF_PROMOS)
                ),
            )
            # Add line through zero
            axis[1].axhline(0, color="k", linewidth=0.5)
            axis[1].set_title("Promo")

            # Enable zoom
            disconnect_zoom = zoom_factory(axis[0])
            disconnect_zoom = zoom_factory(axis[1])

        # Enable pan
        pan_handler = panhandler(fig_2)
        show(block=False)

    def _print_prediction_stacked_graph(self, ssm: StateSpaceModel):
        local_level = sum(
            ssm.parameters.Z[:, 0, i] * ssm.filter.predicted_alpha[:, i, 0]
            for i in range(0, 2)
        )
        ask = sum(
            ssm.parameters.Z[:, 0, i] * ssm.filter.predicted_alpha[:, i, 0]
            for i in range(ms.INDEX_ASK, ms.INDEX_ASK + ms.NUMBER_OF_ASK)
        )
        weekly = sum(
            ssm.parameters.Z[:, 0, i] * ssm.filter.predicted_alpha[:, i, 0]
            for i in range(
                ms.INDEX_W_GAMMA, ms.INDEX_W_GAMMA + ms.NUMBER_OF_WEEKLY_GAMMAS
            )
        )
        beta = sum(
            ssm.parameters.Z[:, 0, i] * ssm.filter.predicted_alpha[:, i, 0]
            for i in range(ms.INDEX_BETA, ms.INDEX_BETA + ms.NUMBER_OF_BETAS)
        )
        promo = sum(
            ssm.parameters.Z[:, 0, i] * ssm.filter.predicted_alpha[:, i, 0]
            for i in range(ms.INDEX_PROMO, ms.INDEX_PROMO + ms.NUMBER_OF_PROMOS)
        )

        y = row_stack((local_level, ask, weekly, beta, promo))
        y_stack = cumsum(y, axis=0)
        x = ssm.parameters.dates

        with ioff():
            fig_1, axis = subplots(1, 1)
            fig_1.set_size_inches(11, 9)
            fig_1.set_tight_layout(True)

            axis.fill_between(x, y_stack[0, :], y2=0, alpha=0.7)
            axis.fill_between(x, y_stack[0, :], y_stack[1, :], alpha=0.7)
            axis.fill_between(x, y_stack[1, :], y_stack[2, :], alpha=0.7)
            axis.fill_between(x, y_stack[2, :], y_stack[3, :], alpha=0.7)
            axis.fill_between(x, y_stack[3, :], y_stack[4, :], alpha=0.7)
            axis.legend(["Local_level", "ASK", "Weekly", "Beta", "Promo"])

            disconnect_zoom = zoom_factory(axis)

        # Enable pan
        pan_handler = panhandler(fig_1)
        show(block=False)

    def _print_scores(self, ssm: StateSpaceModel):
        """Print the values of the different score metrics

        Args:
            ssm (StateSpaceModel): Object containing all the scores
        """

        # Retrieve all the variables from the class and use those to display their values
        score_values = vars(ssm.scores)

        for key in score_values:
            print(f"{key:<15}{score_values[key]:.3f}")

    def _print_prediction_graphs(self, ssm: StateSpaceModel):
        """Print graphs showing information about the prediction and its errors

        Args:
            ssm (StateSpaceModel): Object the results from the Kalman filter to plot

        """
        # Make two subplots and set the figure size
        with ioff():
            fig, axis = subplots(2, 1)
            fig.set_size_inches(12, 9)

            # Plot for data and filtered state vector
            axis[0].plot(ssm.parameters.dates, ssm.parameters.y, "b-")
            axis[0].plot(
                ssm.parameters.dates,
                (ssm.parameters.Z @ ssm.filter.predicted_alpha)[:, 0, 0],
                "r--",
            )
            axis[0].set_title("Observations and predictions")
            axis[0].legend(["y", "Predicted"])

            axis[1].plot(ssm.parameters.dates, ssm.filter.e[:, 0, 0])
            axis[1].plot(ssm.parameters.dates, ssm.filter.v[:, 0, 0])
            axis[1].set_title("forecast errors")
            axis[1].legend(["standardized forecast error", "real forecast error"])

            # Add vertical line for blakc friday in both plots
            for idx, value in enumerate(ssm.parameters.Z[:, 0, ms.INDEX_PROMO]):
                if value == 1:
                    axis[0].axvline(
                        x=ssm.parameters.dates.iloc[idx],
                        color="r",
                        linestyle="--",
                        label="Black friday",
                        linewidth=0.2,
                    )
                    axis[1].axvline(
                        x=ssm.parameters.dates.iloc[idx],
                        color="r",
                        linestyle="--",
                        label="Black friday",
                        linewidth=0.2,
                    )

            # Add vertical line for blakc friday for both plots
            for idx, value in enumerate(ssm.parameters.Z[:, 0, ms.INDEX_PROMO + 1]):
                if value == 1:
                    axis[0].axvline(
                        x=ssm.parameters.dates.iloc[idx],
                        color="b",
                        linestyle="--",
                        label="Cyber Latam",
                        linewidth=0.2,
                    )
                    axis[1].axvline(
                        x=ssm.parameters.dates.iloc[idx],
                        color="b",
                        linestyle="--",
                        label="Cyber Latam",
                        linewidth=0.2,
                    )

            fig.suptitle("Prediction and errors", fontsize=16)
            disconnect_zoom = zoom_factory(axis[0])
            disconnect_zoom = zoom_factory(axis[1])

        # Enable pan
        pan_handler = panhandler(fig)
        show(block=False)

    def _print_forecast_graphs(self, ssm: StateSpaceModel):
        """Print graphs showing information about the kalman filter output

        Args:
            ssm (StateSpaceModel): Object the results from the Kalman filter to plot

        """
        # Make four subplots and set the figure size
        fig, axis = subplots(2, 1)
        fig.set_size_inches(9, 7)

        # Make x-axis variables
        x = array(range(0, len(ssm.parameters.y) + ssm.forecaster.forecast_time_steps))

        # Plot for data and filtered state vector
        axis[0].plot(
            x[-(ssm.forecaster.forecast_time_steps + 10) :],
            ssm.forecaster.complete_y[-(ssm.forecaster.forecast_time_steps + 10) :],
            "b-",
        )
        axis[0].plot(
            x[-(ssm.forecaster.forecast_time_steps + 10) :],
            ssm.forecaster.complete_lower_bound[
                -(ssm.forecaster.forecast_time_steps + 10) :, 0, 0
            ],
            "k",
            linewidth=0.5,
        )
        axis[0].plot(
            x[-(ssm.forecaster.forecast_time_steps + 10) :],
            ssm.forecaster.complete_upper_bound[
                -(ssm.forecaster.forecast_time_steps + 10) :, 0, 0
            ],
            "k",
            linewidth=0.5,
        )
        axis[0].set_title("y (blue line), confidence (black)")

        # Plot for the filtered variance
        axis[1].plot(
            x[-(ssm.forecaster.forecast_time_steps + 10) :],
            ssm.forecaster.complete_F[
                -(ssm.forecaster.forecast_time_steps + 10) :, 0, 0
            ],
        )
        axis[1].set_title("State variance(Pt)")
        show(block=False)

    def _print_diagnostic_values(self, ssm: StateSpaceModel):
        """Print values of diagnostics to check for normality, heteroscedasticity and serial correlation

        Args:
            ssm (StateSpaceModel): Object containing all the information about the diagnostics
        """
        # Normality
        print(
            f"\r\nNormality (alpha = {DiagnosticTests.SIGNIFICANCE})\n"
            + f"Skewness: {ssm.diagnostics.is_skewed}; with value: {ssm.diagnostics.forecast_error_skewness:.3f}\n"
            + f"Kurtosis: {ssm.diagnostics.is_kurtosised}; with value: {ssm.diagnostics.forecast_error_kurtosis:.3f}"
        )

        # Heteroscedasticity
        print(
            f"\nHeteroscedasticity (alpha = {DiagnosticTests.SIGNIFICANCE}, sample size = {DiagnosticTests.SUBSAMPLE_SIZE})\n"
            + f"Homoscedasticity: {ssm.diagnostics.is_homosced}; with value: {ssm.diagnostics.heterosced_test_value:.3f}, "
        )

        # TODO fix auto correlation diagnostics
        # # Serial correlation
        # print(
        #     f"\nSerial correlation (alpha = {DiagnosticTests.SIGNIFICANCE}, lags = {DiagnosticTests.LAGS})\n"
        #     + self._get_serial_correlated_lags_text(ssm)
        # )

    def _get_serial_correlated_lags_text(self, ssm: StateSpaceModel):
        """Formats text to show which lags are correlated"""
        correlated_lag_text = ""
        # Check if all values are false
        if (
            ssm.diagnostics.serial_correlation_test_values["is_serial_correlated"]
            .eq(False)
            .all()
        ):
            correlated_lag_text += "No lags are correlated"
            return correlated_lag_text

        # If not all values are false, append which one are true
        for index, row in ssm.diagnostics.serial_correlation_test_values.iterrows():
            if row["is_serial_correlated"]:
                correlated_lag_text += f"Lag {index} is serial correlated with p-value {row['lb_pvalue']:.4f}\n"
        return correlated_lag_text

    def _print_diagnostic_graphs(self, ssm: StateSpaceModel):
        # Make four subplots and set the figure size
        fig, axis = subplots(2, 2)
        fig.set_size_inches(12, 10)

        # Make x-axis variables
        x = array(range(0, len(ssm.diagnostics.forecast_errors)))

        # Plot for Standardize forecast errors
        axis[0, 0].plot(x, ssm.diagnostics.forecast_errors, "b-")
        axis[0, 0].set_title("Standardize forecast errors (blue line)")

        # Histogram for Standardize forecast errors
        histplot(ssm.diagnostics.forecast_errors, ax=axis[0, 1], kde=True)
        axis[0, 1].set_title("Histogram of standardized forecast errors and density")

        # QQ plot
        qqplot(ssm.diagnostics.forecast_errors, ax=axis[1, 0], line="45")
        axis[1, 0].set_title("QQ-plot of standardized forecast errors")

        # Plot for the prediction variance Ft. not quite sure where used for but was also in the book
        plot_pacf(
            ssm.diagnostics.forecast_errors,
            ax=axis[1, 1],
            lags=DiagnosticTests.LAGS,
        )
        axis[1, 1].set_title("Partial auto correlation of standardized forecast errors")
        show(block=False)

    def _check_errors(self, ssm: StateSpaceModel):
        dates = ssm.parameters.dates.to_numpy()
        errors = ssm.filter.v[:, 0, 0]
        df = DataFrame()
        df["DATES"] = dates
        df["ERRORS"] = abs(errors)
        sorted_df = df.sort_values(by="ERRORS", ascending=False)
        print(sorted_df.head(int(len(sorted_df) * 0.1)))
