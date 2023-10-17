from numpy import ndarray, array, identity, cos
from ssm_model.constants import ModelSpecifications as ms
from ssm_model.constants import InitialKalmanFilter as ikf
from helpers import (
    empty_vector,
    empty_3d_array,
    lamb,
    diagonally_append,
    C_matrix,
    apply_marketing_transformation,
    zeros,
)
from ssm_model.estimated_parameters import EstimatedParameters
from pandas import DataFrame, Series


class StateSpaceParameters:
    """
    State space matrices and variables
    p = number of observations, each observation has its own variance
    m = number of variables in state
    r = number variances in state equation
    """

    y: ndarray  # t*[px1] List with observations
    Z: ndarray  # t*[pxm] Variables that influence y
    H: ndarray  # [pxp] Variance of the observation equation: eps(t) ~ N(0, H)
    T: ndarray  # [mxm] Transition matrix
    R: ndarray  # [mxr] Matrix influencing the state equation variance
    Q: ndarray  # [rxr] variance of the state equation:  eta(t) ~ N(0, Q)

    estimated_parameters: EstimatedParameters  # Object containing all the estimated parameters

    starting_alpha: ndarray  # [mx1] alpha_0, starting state
    starting_P: ndarray  # [mxM] P_0, starting state variance

    dates: Series  # List of dates for visualization

    def __init__(self, y):
        self.y = y
        self.estimated_parameters = EstimatedParameters(False)

    def set_matrices(
        self,
        estimated_parameters: ndarray,
        marketing_data: DataFrame,
        starting_alpha: ndarray = None,
        starting_P: ndarray = None,
    ):
        """Put the variables you want to estimate in the correct matrices
        Also set all other matrices depending on if it is the test or train model

        Args:
            estimated_parameters (ndarray): Array of all the parameters you want to estimate
            marketing_data (DataFrame): Data with the marketing and promo data (and ASK). Contains both the train and test, is split in the _set_Z method
            starting_alpha (ndarray): if given, is the last predicted alpha of the train data, is set as the first of the test
            starting_P (ndarray): if given, is the last predicted P of the train data, is set as the first of the test
        """
        self.estimated_parameters.set_via_array(estimated_parameters)
        self._set_HTRQ_matrices()

        # Starting alpha, use the original parameters if not specifically given
        if starting_alpha is not None:
            # If a starting alpha is given you are using the test set thus set the Z-matrix accordingly
            self._set_Z(marketing_data, False)
            self.starting_alpha = starting_alpha
        else:
            self._set_Z(marketing_data, True)
            self.starting_alpha = array(
                [
                    [
                        self.estimated_parameters.start_trend,
                        0,
                        self.estimated_parameters.beta_1,
                        self.estimated_parameters.beta_2,
                        self.estimated_parameters.beta_3,
                        self.estimated_parameters.beta_4,
                        self.estimated_parameters.beta_5,
                        self.estimated_parameters.beta_6,
                        self.estimated_parameters.beta_7,
                        self.estimated_parameters.beta_8,
                        self.estimated_parameters.promo_1,
                        self.estimated_parameters.promo_2,
                        self.estimated_parameters.wgamma_1,
                        self.estimated_parameters.wgamma_1_star,
                        self.estimated_parameters.wgamma_2,
                        self.estimated_parameters.wgamma_2_star,
                        self.estimated_parameters.wgamma_3,
                        self.estimated_parameters.wgamma_3_star,
                        self.estimated_parameters.ygamma_1,
                        self.estimated_parameters.ygamma_1_star,
                        self.estimated_parameters.ygamma_2,
                        self.estimated_parameters.ygamma_2_star,
                        self.estimated_parameters.ygamma_3,
                        self.estimated_parameters.ygamma_3_star,
                        self.estimated_parameters.ygamma_4,
                        self.estimated_parameters.ygamma_4_star,
                        self.estimated_parameters.ygamma_5,
                        self.estimated_parameters.ygamma_5_star,
                        self.estimated_parameters.ygamma_6,
                        self.estimated_parameters.ygamma_6_star,
                    ]
                ]
            ).T

        # Starting P
        if starting_P is not None:
            self.starting_P = starting_P
        else:
            self.starting_P = identity(ms.NUMBER_OF_STATE_PARAMETERS) * ikf.VARIANCE
            # All variance are equal to the ikf.VARIANCE except for the exogenous variable which are the betas, ask and promo rows
            for i in range(2, 2 + ms.NUMBER_OF_BETAS + ms.NUMBER_OF_PROMOS):
                self.starting_P[i, i] = 0

    def _set_Z(self, marketing_data: DataFrame, is_train: bool):
        """Creates the Z-matrix which contains the marketing data.
        The complete (train+ test) is passed in since we need the max value of the total for the saturation curve

        Args:
            marketing_data (DataFrame): Complete dataframe with both test and train set
            is_train (bool): Return Z for train or test dependent on this bool
        """
        Z_matrix = empty_3d_array(len(marketing_data), 1, ms.NUMBER_OF_STATE_PARAMETERS)
        # LL
        Z_matrix[:, 0, 0:2] = array([[1, 0]])
        # exo
        # get the needed variables and apply the marketing transformation
        for i in range(1, ms.NUMBER_OF_BETAS + 1):
            channel = f"Channel_{i}"
            theta = getattr(self.estimated_parameters, f"theta_{i}")
            phi = getattr(self.estimated_parameters, f"phi_{i}")
            rho = getattr(self.estimated_parameters, f"rho_{i}")

            Z_matrix[:, 0, i + 1] = apply_marketing_transformation(
                marketing_data[channel],
                theta,
                phi,
                rho,
            )
        # Promo
        Z_matrix[:, 0, ms.INDEX_PROMO] = marketing_data["Promo_1"]
        Z_matrix[:, 0, ms.INDEX_PROMO + 1] = marketing_data["Promo_2"]

        # weekly season
        Z_matrix[
            :, 0, ms.INDEX_W_GAMMA : ms.INDEX_W_GAMMA + ms.NUMBER_OF_WEEKLY_GAMMAS
        ] = array([[1, 0, 1, 0, 1, 0]])

        # yearly season
        Z_matrix[
            :, 0, ms.INDEX_Y_GAMMA : ms.INDEX_Y_GAMMA + ms.NUMBER_OF_YEARLY_GAMMAS
        ] = array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]])

        # Split data into train and test
        train_size = int(round(ms.TRAIN_SIZE_PERC * len(marketing_data)))

        if is_train:
            self.dates = marketing_data["Date"]
            self.Z = Z_matrix[:train_size]
        else:
            self.dates = marketing_data["Date"].iloc[train_size - 1 :]
            self.Z = Z_matrix[train_size - 1 :]

    def _set_HTRQ_matrices(self):
        """Set the SSM matrices whcih stay the same for the test and train"""
        # H "matrix"
        self.H = self.estimated_parameters.H

        # T matrix
        T_LL = array([[1, 1], [0, 1]])
        T_exo = identity(ms.NUMBER_OF_BETAS)
        T_promo = identity(ms.NUMBER_OF_PROMOS)
        T_wseason = identity(ms.NUMBER_OF_WEEKLY_GAMMAS)
        T_wseason[0:2, 0:2] = C_matrix(lamb(s=ms.WEEKLY_SEASON_SIZE, j=1))  # 1/7
        T_wseason[2:4, 2:4] = C_matrix(lamb(s=ms.WEEKLY_SEASON_SIZE, j=2))  # 2/7
        T_wseason[4:6, 4:6] = C_matrix(lamb(s=ms.WEEKLY_SEASON_SIZE, j=3))  # 3/7

        T_yseason = identity(ms.NUMBER_OF_YEARLY_GAMMAS)
        T_yseason[0:2, 0:2] = C_matrix(lamb(s=ms.YEARLY_SEASON_SIZE, j=1))  # 1/365
        T_yseason[2:4, 2:4] = C_matrix(lamb(s=ms.YEARLY_SEASON_SIZE, j=2))  # 2/365
        T_yseason[4:6, 4:6] = C_matrix(lamb(s=ms.YEARLY_SEASON_SIZE, j=3))  # 3/365
        T_yseason[6:8, 6:8] = C_matrix(lamb(s=ms.YEARLY_SEASON_SIZE, j=4))  # 4/365
        T_yseason[8:10, 8:10] = C_matrix(lamb(s=ms.YEARLY_SEASON_SIZE, j=5))  # 5/365
        T_yseason[10:12, 10:12] = C_matrix(lamb(s=ms.YEARLY_SEASON_SIZE, j=6))  # 6/365

        self.T = diagonally_append(T_LL, T_exo, T_promo, T_wseason, T_yseason)

        # R matrix
        R_LL = array([[1, 0], [0, 1]])
        R_exo = zeros((ms.NUMBER_OF_BETAS, ms.NUMBER_OF_BETAS))
        R_promo = zeros((ms.NUMBER_OF_PROMOS, ms.NUMBER_OF_PROMOS))
        R_wseason = identity(ms.NUMBER_OF_WEEKLY_GAMMAS)
        R_yseason = identity(ms.NUMBER_OF_YEARLY_GAMMAS)
        self.R = diagonally_append(R_LL, R_exo, R_promo, R_wseason, R_yseason)

        # Q matrix
        Q_LL = array(
            [[self.estimated_parameters.Q_mu, 0], [0, self.estimated_parameters.Q_nu]]
        )
        Q_exo = zeros((ms.NUMBER_OF_BETAS, ms.NUMBER_OF_BETAS))
        Q_promo = zeros((ms.NUMBER_OF_PROMOS, ms.NUMBER_OF_PROMOS))
        Q_wseason = self.estimated_parameters.Q_weekly * identity(
            ms.NUMBER_OF_WEEKLY_GAMMAS
        )
        Q_yseason = self.estimated_parameters.Q_yearly * identity(
            ms.NUMBER_OF_YEARLY_GAMMAS
        )

        self.Q = diagonally_append(Q_LL, Q_exo, Q_promo, Q_wseason, Q_yseason)
