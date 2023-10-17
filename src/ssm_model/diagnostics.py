from ssm_model.kalman_filter import KalmanFilter
from ssm_model.constants import (
    SteadyState,
    DiagnosticTests,
    ModelSpecifications,
)
from numpy import mean, var, ndarray
from scipy.stats import skew, kurtosis, norm, f
from statsmodels.stats.diagnostic import acorr_ljungbox
from pandas import DataFrame, Series


class Diagnostics:
    # General
    size: int  # number of observations, same as n
    forecast_errors: ndarray  # standardized forecast errors used in the diagnostics

    # Steady statte
    steady_state: int  # At which time step the steady state was reached

    # Normality
    forecast_error_mean: float  # First moment of forecast errors
    forecast_error_variance: float  # Second moment of forecast errors
    forecast_error_skewness: float  # Third moment of forecast errors
    forecast_error_kurtosis: float  # Fourth moment of forecast errors
    combined_skew_kurtosis: float  # Combines third and fourth moment
    is_skewed: bool  # True if data has a too high skewness
    is_kurtosised: bool  # True if data has a too high kurtosis
    is_normal: bool  # True if data is normal

    # Heteroscedasticity
    heterosced_test_value: float  # Value obtained from the Heteroscedasticity test
    is_homosced: bool  # True if errors are Homoscedasticity

    # Serial correlation
    serial_correlation_test_values: DataFrame  # Value obtained of the ljungbox test
    is_serial_correlated: Series  # True if errors are serial correlated

    def __init__(self, size: int):
        self.size = size

    def set_diagnostics(self, filter: KalmanFilter):
        """Sets all diagnostic values

        Args:
            filter (KalmanFilter): Object containing the standardized one-step ahead forecast error.
        """
        # Throw away the first standardized forecast error since we used diffuse initialization
        self.forecast_errors = filter.e[:, 0, 0]
        # self._set_steady_state(filter)

        # Normality check
        self._test_normality()

        # Heteroskedasticity test
        self._test_heteroskedasticity(DiagnosticTests.SUBSAMPLE_SIZE)

        # Serial correlation tests
        self._test_serial_correlation(DiagnosticTests.LAGS)

    def _set_steady_state(self, filter: KalmanFilter):
        """Finds when steady state is reached

        Args:
            filter (KalmanFilter): contains the predicted state variance
        """
        t = 0
        while (
            filter.predicted_P[t, 0, 0] - filter.predicted_P[t + 1, 0, 0]
            > SteadyState.THRESHOLD
        ):
            t += 1
        self.steady_state = t

    def _test_normality(self):
        # Set the first 4 moments
        self._set_moments()

        # Check if errors are skewed or kurtosised
        self._check_moments()

        # Errors are normal if both not skewed and not kurtosised
        self.is_normal = self.is_skewed is False and self.is_kurtosised is False

    def _set_moments(self):
        """Set the first 4 moments of the Standardized one-step ahead forecast errors

        Args:
            filter (KalmanFilter): Object that contains the standardized one-step ahead forecast error.
        """
        self.forecast_error_mean = mean(self.forecast_errors)
        self.forecast_error_variance = var(self.forecast_errors)

        # TODO: check if you need to set the property bias=TRUE or FALSE
        self.forecast_error_skewness = skew(self.forecast_errors)
        self.forecast_error_kurtosis = kurtosis(self.forecast_errors, fisher=False)
        self.combined_skew_kurtosis = self.size * (
            pow(self.forecast_error_skewness, 2) / 6
            + pow((self.forecast_error_kurtosis - 3), 2) / 24
        )

    def _check_moments(self):
        """Normality check with the skewness and kurtosis"""

        # check if skewness value lays in the 95% of N(0, 6/n)
        self.is_skewed = (
            not DiagnosticTests.SIGNIFICANCE / 2
            < norm(loc=0, scale=(6 / self.size) ** 0.5).cdf(
                self.forecast_error_skewness
            )
            < 1 - DiagnosticTests.SIGNIFICANCE / 2
        )

        # check if kurtosis value lays in the 95% of N(3, 24/n)
        self.is_kurtosised = (
            not DiagnosticTests.SIGNIFICANCE / 2
            < norm(loc=3, scale=(24 / self.size) ** 0.5).cdf(
                self.forecast_error_kurtosis
            )
            < 1 - DiagnosticTests.SIGNIFICANCE / 2
        )

    def _test_heteroskedasticity(self, size_subset):
        self._set_heteroskedasticity_value(size_subset)

        # TODO: figure out if this is correct
        # Checks if the found f-statistic lays in the 95% confidence interval of an f-distribution
        self.is_homosced = (
            DiagnosticTests.SIGNIFICANCE
            < f(size_subset, size_subset).cdf(self.heterosced_test_value)
            < 1 - DiagnosticTests.SIGNIFICANCE
        )

    def _set_heteroskedasticity_value(self, size_subset: int):
        """Split the forecast errors into two, square and sum each part and divide by eachother
        Value is used to determine if data is homoskedastic

        Args:
            size_subset (int): How big the tested subset is
        """
        # Split the array into two equal parts of size_subset
        h_sized_array = self.forecast_errors[:size_subset]
        # Gets the last size_subset entries from the array
        other_array = self.forecast_errors[-size_subset:]

        # Divide the sqaured-summed other array by the squared-summed h array
        self.heterosced_test_value = self._square_and_sum(
            other_array
        ) / self._square_and_sum(h_sized_array)

    def _test_serial_correlation(self, lags: int):
        """Perform a ljung box test to test serial correlation

        Args:
            lags (int): _description_
        """
        self.serial_correlation_test_values = acorr_ljungbox(
            self.forecast_errors,
            lags,
        )

        # If p-value (lb_pvalue) is smaller than the significance we accept H_a and thus errors are serial correlated
        self.is_serial_correlated = (
            self.serial_correlation_test_values["lb_pvalue"]
            < DiagnosticTests.SIGNIFICANCE
        )

        # Also save column to the test value dataframe to make it easier for displaying
        self.serial_correlation_test_values[
            "is_serial_correlated"
        ] = self.is_serial_correlated

    def _square_and_sum(self, arr: ndarray) -> float:
        """First squares and then sums an array"""
        squared_arr = [x**2 for x in arr]
        return sum(squared_arr)
