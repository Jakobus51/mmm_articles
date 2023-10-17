from ssm_model.kalman_filter import KalmanFilter
from ssm_model.state_space_parameters import StateSpaceParameters
from numpy import log, pi, abs, ndarray, isnan, matrix, mean, diff, divide, power
from ssm_model.constants import ModelSpecifications as ms
from pandas import DataFrame


class Scores:
    # General
    size: int  # number of observations, same as n

    # Information criteria
    LL_score: float  # Summed LL
    AIC: float  # Akaike's information criteria . AIC = 2*k - 2* LL
    BIC: float  # Bayesian information criteria. BIC = 2*k - ln(n)*LL

    # Fit criteria's
    NRMSE: float  # Normalized root mean squared error
    MAE: float  # Mean absolute error. MAE = mean(abs(v)) with v= forecast errors
    MASE: float  # Mean absolute scaled error. MASE = MAE/ MAE_{naive}
    MAPE: float  # Mean absolute percentage error. MAPE = mean(abs(v/y)) with v= forecast errors, y= real value
    R_squared: float
    adj_R_squared: float

    def __init__(self, size: int):
        self.size = size

    def set_scores(self, filter: KalmanFilter, y: ndarray, predictions: ndarray):
        """Sets the LL-score and the information criteria's

        Args:
            filter (KalmanFilter): _description_
        """
        self._set_LL(filter)
        self._set_information_criterias()
        self._set_fit_metrics(y, predictions)

    def _set_LL(self, filter: KalmanFilter):
        """Sets the Log likelihood score

        Args:
            filter (KalmanFilter): Object which contains information needed for the LL score
        """

        self.LL_score = -sum(
            -0.5 * log(2 * pi)
            - 0.5 * (log(abs(filter.F)) + filter.v * pow(filter.F, -1) * filter.v)
        )[0, 0]

    def _set_information_criterias(self):
        """Sets the AIc and BIC scores"""
        self.AIC = 2 * self.LL_score + ms.ESTIMATED_PARAMTERS * 2
        self.BIC = 2 * self.LL_score + ms.ESTIMATED_PARAMTERS * log(self.size)

    def _set_fit_metrics(self, y: ndarray, predictions: ndarray):
        """Set the fit metrics

        Args:
            y (ndarray): True values
            predictions (ndarray): Predictions
        """
        MAE, MASE, MAPE, NRMSE, R_squared, adj_R_squared = get_scores(
            y, predictions, ms.ESTIMATED_PARAMTERS
        )
        self.MAE = MAE
        self.MASE = MASE
        self.MAPE = MAPE
        self.NRMSE = NRMSE
        self.R_squared = R_squared
        self.adj_R_squared = adj_R_squared


def LL_optimization_function(
    variables: ndarray, y: ndarray, marketing_data: DataFrame
) -> float:
    """Returns the logLikelihood for the given T, H and Q on y
    Needs to be this format to work with scipy

    Args:
        variables (tuple): Tuple wherein T, H and Q are stored
        y (ndarray): ndarray with observations

    Returns:
        float: Log likelihood score
    """

    # Save the input variables to a local ssp instance
    ssp = StateSpaceParameters(y)
    ssp.set_matrices(variables, marketing_data)

    # create and run the filter using the given parameters
    kalman_filter = KalmanFilter(len(y))
    kalman_filter.run_kalman_filter(ssp)

    scores = Scores(len(y))
    scores._set_LL(kalman_filter)

    # print(f"{scores.LL_score:.1f}")

    if isnan(scores.LL_score):
        raise ValueError("Invalid LL score")

    return scores.LL_score


def print_scores(true: ndarray, prediction: ndarray, parameter_penalty: int):
    """Print the different fit metrics

    Args:
        true (ndarray): Observations
        prediction (ndarray): Predictions
    """
    MAE, MASE, MAPE, NRMSE, R_squared, adj_R_squared = get_scores(
        true, prediction, parameter_penalty
    )

    print(f"{'MAE':<15}{MAE:.3f}")
    print(f"{'MASE':<15}{MASE:.3f}")
    print(f"{'MAPE':<15}{MAPE:.3f}")
    print(f"{'NRMSE':<15}{NRMSE:.3f}")
    print(f"{'R_squared':<15}{R_squared:.3f}")
    print(f"{'adj_R_squared':<15}{adj_R_squared:.3f}")


def get_scores(true: ndarray, prediction: ndarray, parameter_penalty: int) -> tuple():
    """Print the different fit metrics

    Args:
        true (ndarray): Observations
        prediction (ndarray): Predictions

    Returns:
        tuple(): MAE, MASE, MAPE, NRMSE, R_squared, adj_R_squared
    """

    MAE = mean(abs(true - prediction))

    # MASE (Mean Absolute Scaled Error)
    MAE_naive = mean(abs(diff(true)))  # MAE of the y_i - y_{i-1}
    MASE = MAE / MAE_naive

    # MAPE (Mean Absolute Percentage Error)
    MAPE = mean(abs(divide(true - prediction, true)))

    # NRMSE Normalized Root Mean Squared Error
    RMSE = power((mean(power(true - prediction, 2))), 0.5)
    NRMSE = RMSE / (max(true) - min(true))

    # R_squared
    SS_res = sum(power(true - prediction, 2))
    SS_tot = sum(power((true - mean(true)), 2))
    R_squared = 1 - SS_res / SS_tot

    # Adjusted R_squared
    adj_R_squared = 1 - ((1 - R_squared) * (len(true) - 1)) / (
        len(true) - parameter_penalty - 1
    )

    return (MAE, MASE, MAPE, NRMSE, R_squared, adj_R_squared)
