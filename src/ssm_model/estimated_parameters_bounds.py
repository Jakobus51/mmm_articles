from numpy import ndarray


class EstimatedParametersBounds:
    """Class which contains the bounds of each estimated parameter"""

    def __init__(self):
        # Set standard bounds

        self.start_trend_bound = (None, None)
        # betas
        self.beta_1_bound = (0, None)
        self.beta_2_bound = (0, None)
        self.beta_3_bound = (0, None)
        self.beta_4_bound = (0, None)
        self.beta_5_bound = (0, None)
        self.beta_6_bound = (0, None)
        self.beta_7_bound = (0, None)
        self.beta_8_bound = (0, None)

        # Adstock
        self.theta_1_bound = (0, 0.9)
        self.theta_2_bound = (0, 0.9)
        self.theta_3_bound = (0, 0.9)
        self.theta_4_bound = (0, 0.9)
        self.theta_5_bound = (0, 0.9)
        self.theta_6_bound = (0, 0.9)
        self.theta_7_bound = (0, 0.9)
        self.theta_8_bound = (0, 0.9)

        # Curve shape
        self.phi_1_bound = (0.5, 3)
        self.phi_2_bound = (0.5, 3)
        self.phi_3_bound = (0.5, 3)
        self.phi_4_bound = (0.5, 3)
        self.phi_5_bound = (0.5, 3)
        self.phi_6_bound = (0.5, 3)
        self.phi_7_bound = (0.5, 3)
        self.phi_8_bound = (0.5, 3)

        # Inflection point
        self.rho_1_bound = (0.1, 1)
        self.rho_2_bound = (0.1, 1)
        self.rho_3_bound = (0.1, 1)
        self.rho_4_bound = (0.1, 1)
        self.rho_5_bound = (0.1, 1)
        self.rho_6_bound = (0.1, 1)
        self.rho_7_bound = (0.1, 1)
        self.rho_8_bound = (0.1, 1)

        # Promo
        self.promo_1_bound = (0, None)
        self.promo_2_bound = (0, None)

        # Weekly season
        self.wgamma_1_bound = (None, None)
        self.wgamma_1_star_bound = (None, None)
        self.wgamma_2_bound = (None, None)
        self.wgamma_2_star_bound = (None, None)
        self.wgamma_3_bound = (None, None)
        self.wgamma_3_star_bound = (None, None)
        # yearly season
        self.ygamma_1_bound = (None, None)
        self.ygamma_1_star_bound = (None, None)
        self.ygamma_2_bound = (None, None)
        self.ygamma_2_star_bound = (None, None)
        self.ygamma_3_bound = (None, None)
        self.ygamma_3_star_bound = (None, None)
        self.ygamma_4_bound = (None, None)
        self.ygamma_4_star_bound = (None, None)
        self.ygamma_5_bound = (None, None)
        self.ygamma_5_star_bound = (None, None)
        self.ygamma_6_bound = (None, None)
        self.ygamma_6_star_bound = (None, None)
        # Since H and Q are variance they need to be positive
        self.H_bound = (0.0001, None)
        self.Q_mu_bound = (0.0001, None)
        self.Q_nu_bound = (0.0001, None)
        self.Q_weekly_bound = (0.0001, None)
        self.Q_yearly_bound = (0.0001, None)

    def get_bounds(self) -> list[tuple]:
        """Returns the optimization bounds for every variables

        Returns:
            list[tuple]: the list with tuples containing the lower and upper optimization limit  of each variable
        """
        # Vars returns a dictionary of all variables in the class with its value, get the values of that dictionary and return in a list
        return list(vars(self).values())
