from helpers import empty_vector, empty_3d_array
from numpy import ndarray, zeros
from ssm_model.kalman_filter import KalmanFilter
from ssm_model.state_space_parameters import StateSpaceParameters
from ssm_model.constants import ModelSpecifications as ms
from numpy.linalg import inv


class KalmanSmoother:
    time_steps: int  # how many time steps the observations consists of
    # Variables used for smoothing
    smoothed_alpha: ndarray  # t*[mx1] State after smoothing: alphaHat_t = predictedAlpha_t + P_t * r_(t-1)
    smoothed_P: ndarray  # t*[mxm] Variance of state after smoothing: smoothed_P = P_t - P_t* N_(t-1) * P_t
    smoothed_v: ndarray  # t*[px1] prediction error with the smoothed states
    L: ndarray  # t* [mxm] Needed for smoothing: L_t = T_t- K_t * Z_t = H/F_t
    r: ndarray  # t* [mx1] Weighted sum of innovations: r_(t-1) = Z_t' * F_t^-1 * v_t + L_t' * r_t
    N: ndarray  # t* [mxm] Weighted sum of inverse variances of innovations: N_(t-1) = Z_t' * F_t^-1 * Z_t  + L_t' * N_t * L_t

    def __init__(self, time_steps: int):
        """Make empty vectors which are used in the smoothing
        Also set the last value of r and N to zero since these are initial conditions

        Args:
            size (int): Size of vectors, must be same size as observations matrix
        """
        self.time_steps = time_steps
        self.smoothed_alpha = empty_3d_array(
            self.time_steps, ms.NUMBER_OF_STATE_PARAMETERS, 1
        )
        self.smoothed_P = empty_3d_array(
            time_steps, ms.NUMBER_OF_STATE_PARAMETERS, ms.NUMBER_OF_STATE_PARAMETERS
        )
        self.smoothed_v = empty_3d_array(time_steps, 1, 1)
        self.L = empty_3d_array(
            time_steps, ms.NUMBER_OF_STATE_PARAMETERS, ms.NUMBER_OF_STATE_PARAMETERS
        )
        self.r = empty_3d_array(self.time_steps, ms.NUMBER_OF_STATE_PARAMETERS, 1)
        self.N = empty_3d_array(
            time_steps, ms.NUMBER_OF_STATE_PARAMETERS, ms.NUMBER_OF_STATE_PARAMETERS
        )

        # Set initial constraint, which is the last element since backwards recursion
        self.r[-1] = zeros((ms.NUMBER_OF_STATE_PARAMETERS, 1))
        self.N[-1] = zeros(
            (ms.NUMBER_OF_STATE_PARAMETERS, ms.NUMBER_OF_STATE_PARAMETERS)
        )

    def run_smoothing(self, ssp: StateSpaceParameters, kalman_filter: KalmanFilter):
        # Loop from end of array to beginning
        for t in range(self.time_steps - 1, -1, -1):
            # Set L
            self.L[t] = ssp.T - kalman_filter.K[t] @ ssp.Z[t]

            ### Smoothing
            # State
            self.smoothed_alpha[t] = kalman_filter.predicted_alpha[
                t
            ] + kalman_filter.predicted_P[t] @ (
                ssp.Z[t].T @ inv(kalman_filter.F[t]) @ kalman_filter.v[t]
                + self.L[t].T @ self.r[t]
            )

            # Variance
            self.smoothed_P[t] = (
                kalman_filter.predicted_P[t]
                - kalman_filter.predicted_P[t]
                @ (
                    ssp.Z[t].T @ inv(kalman_filter.F[t]) @ ssp.Z[t]
                    + self.L[t].T @ self.N[t] @ self.L[t]
                )
                @ kalman_filter.predicted_P[t]
            )

            # Save smoother error
            self.smoothed_v[t] = ssp.y[t] - ssp.Z[t] @ self.smoothed_alpha[t]

            # Recursion stuff, use a break since otherwise we get out of index (and you unintentionally set the last element again)
            if t == 0:
                break
            self.r[t - 1] = (
                ssp.Z[t].T @ inv(kalman_filter.F[t]) @ kalman_filter.v[t]
                + self.L[t].T @ self.r[t]
            )
            self.N[t - 1] = (
                ssp.Z[t].T @ inv(kalman_filter.F[t]) @ ssp.Z[t]
                + self.L[t].T @ self.N[t] @ self.L[t]
            )
