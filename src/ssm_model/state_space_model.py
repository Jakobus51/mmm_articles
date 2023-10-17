from ssm_model.kalman_filter import KalmanFilter
from ssm_model.kalman_smoother import KalmanSmoother
from ssm_model.state_space_parameters import StateSpaceParameters
from ssm_model.kalman_forecaster import KalmanForecaster
from ssm_model.diagnostics import Diagnostics
from ssm_model.scores import Scores
from ssm_model.scores import LL_optimization_function
from numpy import ndarray
from ssm_model.constants import ModelSpecifications as ms
from scipy.optimize import minimize
from pandas import DataFrame
from time import perf_counter


class StateSpaceModel:
    """Class that is used to save all parts of the model. Model is a state space model as described by Durbin and Koopman:
    y(t)  = Z*alpha(t)   + eps(t)             with eps(t) ~ N(0, H)
    alpha(t+1) = T * alpha(t) + R* eta(t)      with eta(t) ~ N(0, Q)"""

    parameters: StateSpaceParameters  # Object that contains all of the above mentioned matrices and also observation vector y
    filter: KalmanFilter  # Object which will contain the optimal filter results
    smoother: KalmanSmoother  # Object which will contain the smoother results
    scores: Scores  # Object where the LL and information criteria's are saved
    forecaster: KalmanForecaster  # Object used to forecast ahead
    diagnostics: Diagnostics  # Object to keep track of various diagnostics such as Steady state, normality, heteroscedasticity, serial correlation

    def __init__(self, y: ndarray):
        self.parameters = StateSpaceParameters(y)
        self.filter = KalmanFilter(len(y))
        self.smoother = KalmanSmoother(len(y))
        self.scores = Scores(len(y))
        self.forecaster = KalmanForecaster()
        self.diagnostics = Diagnostics(len(y))

    def train_model(
        self,
        initial_values: ndarray,
        bounds: ndarray,
        marketing_data: DataFrame,
        iteration_multiplier: int,
    ):
        """Run the minimization problem to find the optimal parameters

        Args:
            initial_values (ndarray): The starting values for your optimization problem
            bounds (ndarray): The bounds of the estimated variables
            marketing_data (DataFrame): The marketing data used in the model
            iteration_multiplier (int): How many times the model runs times the length of the initial_values


        """
        #

        minimization_result = minimize(
            fun=LL_optimization_function,
            x0=initial_values,
            bounds=bounds,
            args=(self.parameters.y, marketing_data.copy()),
            method="Nelder-Mead",
            options={
                "disp": False,
                "maxiter": len(initial_values) * iteration_multiplier,
            },
        )

        # Save the optimized parameters to the state space model
        self._save_optimal_kalman_filter(minimization_result, marketing_data.copy())

        # Run the smoother
        self.smoother.run_smoothing(self.parameters, self.filter)

        # Save the gammas from smoother instead of filter for seasonality
        self._save_smoother_gammas()

        # Get scores for the run
        predictions = (self.parameters.Z @ self.filter.predicted_alpha)[:, 0, 0]
        self.scores.set_scores(self.filter, self.parameters.y, predictions)

    def test_model(
        self,
        ssm_train_filter: KalmanFilter,
        ssm_train_param: StateSpaceParameters,
        marketing_data: DataFrame,
    ):
        """Use the trained_model to test on the test-set

        Args:
            ssm_train_filter (KalmanFilter): Object containing the starting alpha en P of the test set
            ssm_train_param (StateSpaceParameters): Estimated parameters by the train set
            marketing_data (DataFrame): The marketing data
        """
        # Initialize the test SSM and set its values
        # First date of test is the same as the last of train to make sure the state is handed over correctly
        self.parameters.set_matrices(
            ssm_train_param.estimated_parameters.get_values(),
            marketing_data,
            ssm_train_filter.predicted_alpha[-1],
            ssm_train_filter.predicted_P[-1],
        )
        # Run filter
        self.filter.run_kalman_filter(self.parameters)

        # Get scores for the test set
        predictions_test = (self.parameters.Z @ self.filter.predicted_alpha)[:, 0, 0]
        self.scores.set_scores(self.filter, self.parameters.y, predictions_test)
        self.diagnostics.set_diagnostics(self.filter)

    def _save_optimal_kalman_filter(
        self, minimization_result, marketing_data: DataFrame
    ):
        """Save the estimated parameters from a scipy minimization result and use those to run a kalman filter

        Args:
            minimization_result (?): The scipy object containing the optimized parameters and LL score
        """
        # Save the optimal output
        self.parameters.set_matrices(minimization_result.x, marketing_data)

        # Use the saved optimal parameters to run a kalman filter and save the output
        self.filter.run_kalman_filter(self.parameters)

    def _save_smoother_gammas(self):
        """Save the starting gammas from the smoother instead of the filter"""

        # Weekly season
        self.parameters.estimated_parameters.wgamma_1 = self.smoother.smoothed_alpha[
            0, ms.INDEX_W_GAMMA, 0
        ]
        self.parameters.estimated_parameters.wgamma_1_star = (
            self.smoother.smoothed_alpha[0, ms.INDEX_W_GAMMA + 1, 0]
        )
        self.parameters.estimated_parameters.wgamma_2 = self.smoother.smoothed_alpha[
            0, ms.INDEX_W_GAMMA + 2, 0
        ]
        self.parameters.estimated_parameters.wgamma_2_star = (
            self.smoother.smoothed_alpha[0, ms.INDEX_W_GAMMA + 3, 0]
        )
        self.parameters.estimated_parameters.wgamma_3 = self.smoother.smoothed_alpha[
            0, ms.INDEX_W_GAMMA + 4, 0
        ]
        self.parameters.estimated_parameters.wgamma_3_star = (
            self.smoother.smoothed_alpha[0, ms.INDEX_W_GAMMA + 5, 0]
        )
