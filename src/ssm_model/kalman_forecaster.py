from ssm_model.kalman_filter import KalmanFilter
from ssm_model.state_space_parameters import StateSpaceParameters
from helpers import empty_vector, empty_3d_array
from numpy import matrix, append
from ssm_model.constants import Forecast
from ssm_model.constants import ModelSpecifications as ms


class KalmanForecaster:
    forecast_time_steps: int  # The number of steps you are going to forecast

    # Variables used for forecasting
    forecasted_alpha: matrix  # ft *[mx1]  The predicted state: a_t+1
    forecasted_P: matrix  # ft *[mxm] The variance of the predicted state: P_t+1
    forecasted_v: matrix  # ft *[px1] Prediction error
    forecasted_F: matrix  # ft * [pxp] Variance of prediction
    forecasted_y: matrix  # ft * [px1] Predicted observation
    forecasted_Z: matrix  # ft * [pxm] Predicted exogenous variables
    lower_bound: matrix  # ft *[px1] Lower bound of forecast
    upper_bound: matrix  # ft *[px1] Upper bound of forecast

    # Variables which append the filter and forecast vectors
    complete_alpha: matrix  # (t+ft) *[mx1] Array containing the filter and forecasted state vector
    complete_P: matrix  # (t+ft) *[mxm] Array containing the filter and forecasted state vector variance
    complete_v: matrix  # (t+ft) *[px1] Array containing the filter and forecasted prediction error
    complete_F: matrix  # (t+ft) *[pxp] Array containing the filter and forecasted prediction variance
    complete_y: matrix  # (t+ft) *[px1] Array containing the original y and nan values for the forecasted
    complete_lower_bound: matrix  # (t+ft) *[px1] Array containing the filter and forecasted lower bound
    complete_upper_bound: matrix  # (t+ft) *[px1] Array containing the filter and forecasted upper bound

    def predict_t_steps(
        self,
        ssp: StateSpaceParameters,
        filter: KalmanFilter,
        forecast_time_steps: int,
        forecasted_Z: matrix,
    ):
        """Predict the state forecast_time_steps steps ahead
        Same procedure as dealing with missing observations

        Args:
            ssp (StateSpaceParameters): Parameters used for the model building
            filter (KalmanFilter): Object containing all the Kalman filter information
            size_forecast (int): The amount of steps you want to forecast
        """
        self._initialize(forecast_time_steps, forecasted_Z)

        for t in range(forecast_time_steps):
            # For the first prediction use the last filtered value
            if t < 1:
                self.forecasted_alpha[t] = ssp.T @ filter.filtered_alpha[-1]
                self.forecasted_P[t] = (
                    ssp.T @ filter.filtered_P[-1] @ ssp.T.T + ssp.R @ ssp.Q @ ssp.R.T
                )
                # self.forecasted_v[t] = filter.v[-1] + ssp.R @ ssp.Q @ ssp.R.T

            # For all other use last predicted state
            else:
                self.forecasted_alpha[t] = ssp.T @ self.forecasted_alpha[t - 1]
                self.forecasted_P[t] = (
                    ssp.T @ self.forecasted_P[t - 1] @ ssp.T.T + ssp.R @ ssp.Q @ ssp.R.T
                )
                # self.forecasted_v[t] = (
                #     self.forecasted_v[t - 1] + ssp.R @ ssp.Q @ ssp.R.T
                # )
            self.forecasted_y[t] = self.forecasted_Z[t] @ self.forecasted_alpha[t]
            self.forecasted_F[t] = (
                self.forecasted_Z[t] @ self.forecasted_P[t] @ self.forecasted_Z[t].T
                + ssp.H
            )

        self._set_confidence_interval()
        self._concatenate_forecasted(filter, ssp)

    def _initialize(self, size_forecast: int, forecasted_Z: matrix):
        """Set the size of all vectors

        Args:
            size_forecast (int): The number of steps you want to forecast
            forecasted_Z (matrix): matrix containing the values of the exogenous variables
        """
        self.forecast_time_steps = size_forecast

        # Forecasted vectors
        self.forecasted_alpha = empty_3d_array(
            self.forecast_time_steps, ms.NUMBER_OF_STATE_PARAMETERS, 1
        )
        self.forecasted_P = empty_3d_array(
            self.forecast_time_steps,
            ms.NUMBER_OF_STATE_PARAMETERS,
            ms.NUMBER_OF_STATE_PARAMETERS,
        )
        self.forecasted_v = empty_3d_array(self.forecast_time_steps, 1, 1)
        self.forecasted_F = empty_3d_array(self.forecast_time_steps, 1, 1)
        self.forecasted_y = empty_3d_array(self.forecast_time_steps, 1, 1)
        self.forecasted_Z = forecasted_Z

        self.lower_bound = empty_3d_array(self.forecast_time_steps, 1, 1)
        self.upper_bound = empty_3d_array(self.forecast_time_steps, 1, 1)

    def _set_confidence_interval(self):
        """Create the confidence intervals of the prediction"""
        self.lower_bound = self.forecasted_y - Forecast.PROBABILITY_OF_INCULSION * pow(
            self.forecasted_F, 0.5
        )
        self.upper_bound = self.forecasted_y + Forecast.PROBABILITY_OF_INCULSION * pow(
            self.forecasted_F, 0.5
        )

    def _concatenate_forecasted(self, filter: KalmanFilter, ssp: StateSpaceParameters):
        """Add the forecasted values to the filtered values

        Args:
            filter (KalmanFilter): Object containing the filtered values
            ssp (StateSpaceParameters): Object containing the original observations
        """
        self.complete_alpha = append(filter.filtered_alpha, self.forecasted_alpha, 0)
        self.complete_P = append(filter.filtered_P, self.forecasted_P, 0)
        # self.complete_v = concatenate((filter.v, self.forecasted_v), axis=None)
        self.complete_F = append(filter.F, self.forecasted_F, 0)
        self.complete_lower_bound = append(
            empty_3d_array(filter.time_steps, 1, 1), self.lower_bound, 0
        )
        self.complete_upper_bound = append(
            empty_3d_array(filter.time_steps, 1, 1), self.upper_bound, 0
        )
        self.complete_y = append(ssp.y, self.forecasted_y[:, 0, 0], 0)
