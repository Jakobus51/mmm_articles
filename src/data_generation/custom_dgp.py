from numpy import (
    random,
    ndarray,
    array,
    identity,
    zeros,
)
from helpers import (
    empty_vector,
    empty_3d_array,
    lamb,
    diagonally_append,
    C_matrix,
    apply_marketing_transformation,
)
from ssm_model.constants import ModelSpecifications as ms
from ssm_model.estimated_parameters import EstimatedParameters
from ssm_model.constants import RandomStates
from pandas import read_csv, Series, to_datetime, DataFrame
from data_generation.dg_constants import DataLocations as dl
from data_generation.true_parameters import TrueParameters
from helpers import save_dataset
from matplotlib.pyplot import subplots, show, ioff
from mpl_interactions import panhandler, zoom_factory


class CustomDGP:
    time_steps: int  # How many observations there are
    scale: int  # How big the variance and exogenous variables are

    y: ndarray  # t*[px1] List with observations
    Z: ndarray  # t*[pxm] Variables that influence y
    H: ndarray  # [pxp] Variance of the observation equation: eps(t) ~ N(0, H)
    T: ndarray  # [mxm] Transition matrix
    R: ndarray  # [mxr] Matrix influencing the state equation variance
    Q: ndarray  # [rxr] variance of the state equation:  eta(t) ~ N(0, Q)

    eps: ndarray  # Vector containing the observation noise caused by variance H
    eta: ndarray  # Matrix containing the observation noise caused by variance Q

    alpha: ndarray  # State matrix
    true_parameters: EstimatedParameters  # Object containing all the estimated parameters

    dates: Series

    def __init__(
        self,
        random_state: int,
        marketing_data: DataFrame,
        save_dgp: bool,
        plot_dgp: bool,
    ):
        """Creates a custom dgp, which can be used for the model to estimate

        Args:
            plot_dgp_bool (bool): IF you want to plot the dgp and its components
            random_state (int, optional): Random state used for dgp creation. Defaults to RandomStates.STATE.
        """
        random.seed(random_state)

        self.time_steps = 3 * 365
        self.true_parameters = TrueParameters().true_parameters

        # Z matrix
        self.Z = self._generate_Z(marketing_data)

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

        # H matrix
        self.H = self.true_parameters.H

        # R matrix
        R_LL = array([[1, 0], [0, 1]])
        R_exo = zeros((ms.NUMBER_OF_BETAS, ms.NUMBER_OF_BETAS))
        R_promo = zeros((ms.NUMBER_OF_PROMOS, ms.NUMBER_OF_PROMOS))
        R_wseason = identity(ms.NUMBER_OF_WEEKLY_GAMMAS)
        R_yseason = identity(ms.NUMBER_OF_YEARLY_GAMMAS)
        self.R = diagonally_append(R_LL, R_exo, R_promo, R_wseason, R_yseason)

        # Q matrix
        Q_LL = array([[self.true_parameters.Q_mu, 0], [0, self.true_parameters.Q_nu]])
        Q_exo = zeros((ms.NUMBER_OF_BETAS, ms.NUMBER_OF_BETAS))
        Q_promo = zeros((ms.NUMBER_OF_PROMOS, ms.NUMBER_OF_PROMOS))
        Q_wseason = self.true_parameters.Q_weekly * identity(ms.NUMBER_OF_WEEKLY_GAMMAS)
        Q_yseason = self.true_parameters.Q_yearly * identity(ms.NUMBER_OF_YEARLY_GAMMAS)
        self.Q = diagonally_append(Q_LL, Q_exo, Q_promo, Q_wseason, Q_yseason)

        # Starting alpha
        self.alpha_0 = array(
            [
                [
                    self.true_parameters.start_trend,
                    0,
                    self.true_parameters.beta_1,
                    self.true_parameters.beta_2,
                    self.true_parameters.beta_3,
                    self.true_parameters.beta_4,
                    self.true_parameters.beta_5,
                    self.true_parameters.beta_6,
                    self.true_parameters.beta_7,
                    self.true_parameters.beta_8,
                    self.true_parameters.promo_1,
                    self.true_parameters.promo_2,
                    self.true_parameters.wgamma_1,
                    self.true_parameters.wgamma_1_star,
                    self.true_parameters.wgamma_2,
                    self.true_parameters.wgamma_2_star,
                    self.true_parameters.wgamma_3,
                    self.true_parameters.wgamma_3_star,
                    self.true_parameters.ygamma_1,
                    self.true_parameters.ygamma_1_star,
                    self.true_parameters.ygamma_2,
                    self.true_parameters.ygamma_2_star,
                    self.true_parameters.ygamma_3,
                    self.true_parameters.ygamma_3_star,
                    self.true_parameters.ygamma_4,
                    self.true_parameters.ygamma_4_star,
                    self.true_parameters.ygamma_5,
                    self.true_parameters.ygamma_5_star,
                    self.true_parameters.ygamma_6,
                    self.true_parameters.ygamma_6_star,
                ]
            ]
        ).T

        self.generate_data()

        if save_dgp:
            save_dataset(marketing_data, self.y, random_state)
        if plot_dgp:
            dgp_plot(self)

    def _generate_Z(self, marketing_data: DataFrame) -> ndarray:
        """Generates Z with the help of the fake marketing csv

        Returns:
            ndarray: The simulated Z matrix
        """
        # Import the fake marketing csv

        fake_Z = empty_3d_array(self.time_steps, 1, ms.NUMBER_OF_STATE_PARAMETERS)
        # LL
        fake_Z[:, 0, 0:2] = array([[1, 0]])
        # exo, get values from the fakemarketing csv and apply the adstock and saturation transformation
        fake_Z[:, 0, 2] = apply_marketing_transformation(
            marketing_data["Channel_1"],
            self.true_parameters.theta_1,
            self.true_parameters.phi_1,
            self.true_parameters.rho_1,
        )
        fake_Z[:, 0, 3] = apply_marketing_transformation(
            marketing_data["Channel_2"],
            self.true_parameters.theta_2,
            self.true_parameters.phi_2,
            self.true_parameters.rho_2,
        )
        fake_Z[:, 0, 4] = apply_marketing_transformation(
            marketing_data["Channel_3"],
            self.true_parameters.theta_3,
            self.true_parameters.phi_3,
            self.true_parameters.rho_3,
        )
        fake_Z[:, 0, 5] = apply_marketing_transformation(
            marketing_data["Channel_4"],
            self.true_parameters.theta_4,
            self.true_parameters.phi_4,
            self.true_parameters.rho_4,
        )
        fake_Z[:, 0, 6] = apply_marketing_transformation(
            marketing_data["Channel_5"],
            self.true_parameters.theta_5,
            self.true_parameters.phi_5,
            self.true_parameters.rho_5,
        )
        fake_Z[:, 0, 7] = apply_marketing_transformation(
            marketing_data["Channel_6"],
            self.true_parameters.theta_6,
            self.true_parameters.phi_6,
            self.true_parameters.rho_6,
        )
        fake_Z[:, 0, 8] = apply_marketing_transformation(
            marketing_data["Channel_7"],
            self.true_parameters.theta_7,
            self.true_parameters.phi_7,
            self.true_parameters.rho_7,
        )
        fake_Z[:, 0, 9] = apply_marketing_transformation(
            marketing_data["Channel_8"],
            self.true_parameters.theta_8,
            self.true_parameters.phi_8,
            self.true_parameters.rho_8,
        )
        # promo, get the promo columns from the fake marketing csv
        fake_Z[:, 0, 10] = marketing_data["Promo_1"]
        fake_Z[:, 0, 11] = marketing_data["Promo_2"]

        # weekly season
        fake_Z[
            :, 0, ms.INDEX_W_GAMMA : ms.INDEX_W_GAMMA + ms.NUMBER_OF_WEEKLY_GAMMAS
        ] = array([[1, 0, 1, 0, 1, 0]])

        # yearly season
        fake_Z[
            :, 0, ms.INDEX_Y_GAMMA : ms.INDEX_Y_GAMMA + ms.NUMBER_OF_YEARLY_GAMMAS
        ] = array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]])

        self.dates = to_datetime(marketing_data["Date"])
        return fake_Z

    def generate_data(self):
        """
        y(t)  = Z(t)*alpha(t)   + eps(t)             with eps(t) ~ N(0, H)

        alpha(t+1) = T * alpha(t) + R* eta(t)      with eta(t) ~ N(0, Q)
        """

        self.alpha = empty_3d_array(self.time_steps, ms.NUMBER_OF_STATE_PARAMETERS, 1)
        self.y = empty_vector(self.time_steps)
        self.eps = empty_vector(self.time_steps)
        self.eta = empty_3d_array(self.time_steps, ms.NUMBER_OF_STATE_VARIANCES, 1)
        for t in range(self.time_steps):
            # Set and save noise
            self.eps[t] = random.normal(0, self.H**0.5)
            self.eta[t] = random.multivariate_normal(
                zeros(ms.NUMBER_OF_STATE_VARIANCES), self.Q**0.5, 1
            ).T

            if t < 1:
                # initialize the first entry of the state vector and observation
                self.alpha[0] = self.alpha_0
                self.y[0] = self.Z[0] @ self.alpha[0] + self.eps[0]
            else:
                self.alpha[t] = self.T @ self.alpha[t - 1] + self.R @ self.eta[t - 1]
                self.y[t] = self.Z[t] @ self.alpha[t] + self.eps[t]


def dgp_plot(dgp: CustomDGP):
    """Plots of the signal
    0. LL (trend)
    1. Beta
    2. weekly
    3. yearly
    4. promo

    Args:
        ssm (StateSpaceModel): _description_
    """
    # First 4 plots
    line_width = 1.5
    with ioff():
        fig_1, axis = subplots(5, 1)
        fig_1.set_size_inches(12, 9)
        fig_1.set_tight_layout(True)
        # LL, trend
        axis[0].plot(
            dgp.dates,
            sum(dgp.alpha[:, i, 0] for i in range(0, 2)),
            linewidth=line_width,
        )
        axis[0].axhline(0, color="k", linewidth=0.5)

        axis[0].set_title("1.Trend")

        # Yearly
        axis[1].plot(
            dgp.dates,
            sum(
                dgp.Z[:, 0, i] * dgp.alpha[:, i, 0]
                for i in range(
                    ms.INDEX_Y_GAMMA, ms.INDEX_Y_GAMMA + ms.NUMBER_OF_YEARLY_GAMMAS
                )
            ),
            linewidth=line_width,
        )
        axis[1].axhline(0, color="k", linewidth=0.5)

        axis[1].set_title("2. Yearly seasonality")

        # Weekly
        axis[2].plot(
            dgp.dates[:71],
            sum(
                dgp.Z[:, 0, i] * dgp.alpha[:, i, 0]
                for i in range(
                    ms.INDEX_W_GAMMA, ms.INDEX_W_GAMMA + ms.NUMBER_OF_WEEKLY_GAMMAS
                )
            )[:71],
            linewidth=line_width,
        )
        axis[2].axhline(0, color="k", linewidth=0.5)
        axis[2].set_title("3. Weekly seasonality (has different x-axis)")

        # Beta
        axis[3].plot(
            dgp.dates[1:],
            sum(
                dgp.Z[:, 0, i] * dgp.alpha[:, i, 0]
                for i in range(ms.INDEX_BETA, ms.INDEX_BETA + ms.NUMBER_OF_BETAS)
            )[1:],
            linewidth=line_width,
        )

        axis[3].set_title("4. Marketing effects")

        # Promo
        axis[4].plot(
            dgp.dates,
            sum(
                dgp.Z[:, 0, i] * dgp.alpha[:, i, 0]
                for i in range(ms.INDEX_PROMO, ms.INDEX_PROMO + ms.NUMBER_OF_PROMOS)
            ),
            linewidth=line_width,
        )
        axis[4].set_title("5. Promotional effects")

        for ax in axis:
            ax.set_ylabel("Revenue")

        # Enable zoom
        disconnect_zoom = zoom_factory(axis[0])
        disconnect_zoom = zoom_factory(axis[1])
        disconnect_zoom = zoom_factory(axis[2])
        disconnect_zoom = zoom_factory(axis[3])
        disconnect_zoom = zoom_factory(axis[4])

        fig_2, axis_2 = subplots()
        fig_2.set_size_inches(12, 9)
        fig_2.set_tight_layout(True)
        # LL, trend
        axis_2.plot(dgp.dates, dgp.y, linewidth=line_width)
        axis_2.axhline(0, color="k", linewidth=0.5)

        axis_2.set_title("Simulated revenue")
        axis_2.set_ylabel("Revenue")

    # Enable pan
    pan_handler = panhandler(fig_1)
    pan_handler = panhandler(fig_2)

    show(block=True)
