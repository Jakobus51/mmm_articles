from ssm_model.constants import ModelSpecifications as ms
from ssm_model.state_space_parameters import StateSpaceParameters
from helpers import empty_vector, empty_3d_array
from numpy import log, pi, abs, ndarray, isnan, sqrt, array
from numpy.linalg import inv


class KalmanFilter:
    """Documentation"""

    time_steps: int  # how many time steps the observations consists of

    # Variables used for filtering
    predicted_alpha: ndarray  # t*[mx1] The predicted state: a_t+1
    predicted_P: ndarray  # t* [mxm] The variance of the predicted state: P_t+1
    filtered_alpha: ndarray  # t*[mx1] Filtered state: a_t|t
    filtered_P: ndarray  # t*[mxm] Variance of filtered state: P_t|t
    v: ndarray  # t*[px1] Prediction error: v_t = y_t - Z * a_t
    F: ndarray  # t*[pxp] Variance of prediction: F_t = Z * P_t * Z' + H
    K: ndarray  # t*[mxp] Kalman gain: K_t = T* P_t * Z' * F_t^-1
    e: ndarray  # t*[px1] Standardized one-step ahead forecast error. e_t = v_t / F_t**0.5

    def __init__(self, time_steps: int):
        """Kalman filter

        Args:
            time_steps (int): _description_
        """

        self.time_steps = time_steps
        self.predicted_alpha = empty_3d_array(
            self.time_steps, ms.NUMBER_OF_STATE_PARAMETERS, 1
        )
        self.predicted_P = empty_3d_array(
            time_steps, ms.NUMBER_OF_STATE_PARAMETERS, ms.NUMBER_OF_STATE_PARAMETERS
        )
        self.filtered_alpha = empty_3d_array(
            self.time_steps, ms.NUMBER_OF_STATE_PARAMETERS, 1
        )
        self.filtered_P = empty_3d_array(
            time_steps, ms.NUMBER_OF_STATE_PARAMETERS, ms.NUMBER_OF_STATE_PARAMETERS
        )
        self.v = empty_3d_array(time_steps, 1, 1)
        self.F = empty_3d_array(time_steps, 1, 1)
        self.K = empty_3d_array(time_steps, ms.NUMBER_OF_STATE_PARAMETERS, 1)
        self.e = empty_3d_array(time_steps, 1, 1)

    def run_kalman_filter(self, ssp: StateSpaceParameters):
        """Retrieves the state vector and its variance as well as all other matrices generated in the filtering part
        Args:
            ssp (StateSpaceParameters): The object containing the observations and matrices used for filtering.
        """
        for t in range(self.time_steps):
            self._predict_kalman(t, ssp)
            self._filter_kalman(t, ssp)
            # Save the KalmanGain and standardized forecast error for usage in other function
            self.K[t] = ssp.T @ self.predicted_P[t] @ ssp.Z[t].T @ inv(self.F[t])
            self.e[t] = self.v[t] @ inv(sqrt(self.F[t]))

    def _predict_kalman(self, t: int, ssp: StateSpaceParameters):
        """Predicts the next state vector and its variance

        Args:
            t (int): time step which you predict
            ssp (StateSpaceParameters): The object containing the observations and matrices used for filtering.
        """
        # Use initial states to set first predicted state and state variance
        if t == 0:
            self.predicted_alpha[0] = ssp.starting_alpha
            self.predicted_P[0] = ssp.starting_P + ssp.R @ ssp.Q @ ssp.R.T
        else:
            self.predicted_alpha[t] = ssp.T @ self.filtered_alpha[t - 1]
            self.predicted_P[t] = (
                ssp.T @ self.filtered_P[t - 1] @ ssp.T.T + ssp.R @ ssp.Q @ ssp.R.T
            )

    def _filter_kalman(self, t: int, ssp: StateSpaceParameters):
        """Update/filters the predicted state vector and its variance with the observed data

        Args:
            t (int): time step which you predict
            ssp (StateSpaceParameters): The object containing the observations and matrices used for filtering.
        """

        # When there are missing values in your observation, set the Z of the local level to zero (page 111)
        # if isnan(ssp.y[t]):
        #     ssp.Z[t, 0, 0] = 0

        # Saving some metrics used in other methods
        # Prediction error
        self.v[t] = ssp.y[t] - ssp.Z[t] @ self.predicted_alpha[t]
        # Prediction variance
        self.F[t] = ssp.Z[t] @ self.predicted_P[t] @ ssp.Z[t].T + ssp.H

        # Filtered state vector
        self.filtered_alpha[t] = (
            self.predicted_alpha[t]
            + self.predicted_P[t] @ ssp.Z[t].T @ inv(self.F[t]) @ self.v[t]
        )

        # Filtered state vector variance
        self.filtered_P[t] = (
            self.predicted_P[t]
            - self.predicted_P[t]
            @ ssp.Z[t].T
            @ inv(self.F[t])
            @ ssp.Z[t]
            @ self.predicted_P[t]
        )
