from numpy import ndarray


class EstimatedParameters:
    """Class which contains all the parameters you want to estimate
    Is also used for the starting values
    """

    def __init__(self, sv: bool):
        # Set standard starting values if requested otherwise set as None

        self.start_trend = 4000 if sv else None
        # exo
        self.beta_1 = 1000 if sv else None
        self.beta_2 = 800 if sv else None
        self.beta_3 = 1000 if sv else None
        self.beta_4 = 200 if sv else None
        self.beta_5 = 750 if sv else None
        self.beta_6 = 800 if sv else None
        self.beta_7 = 400 if sv else None
        self.beta_8 = 1200 if sv else None

        # Adstock
        self.theta_1 = 0.1 if sv else None
        self.theta_2 = 0.8 if sv else None
        self.theta_3 = 0.4 if sv else None
        self.theta_4 = 0.4 if sv else None
        self.theta_5 = 0.8 if sv else None
        self.theta_6 = 0.4 if sv else None
        self.theta_7 = 0.4 if sv else None
        self.theta_8 = 0.4 if sv else None

        # Curve shape
        self.phi_1 = 1 if sv else None
        self.phi_2 = 2 if sv else None
        self.phi_3 = 2 if sv else None
        self.phi_4 = 2 if sv else None
        self.phi_5 = 2 if sv else None
        self.phi_6 = 1 if sv else None
        self.phi_7 = 1 if sv else None
        self.phi_8 = 1 if sv else None

        # Inflection point
        self.rho_1 = 0.3 if sv else None
        self.rho_2 = 0.6 if sv else None
        self.rho_3 = 0.3 if sv else None
        self.rho_4 = 0.6 if sv else None
        self.rho_5 = 0.3 if sv else None
        self.rho_6 = 0.6 if sv else None
        self.rho_7 = 0.6 if sv else None
        self.rho_8 = 0.6 if sv else None

        # promo
        self.promo_1 = 2000 if sv else None
        self.promo_2 = 2000 if sv else None
        # weekly season
        self.wgamma_1 = 100 if sv else None
        self.wgamma_1_star = -100 if sv else None
        self.wgamma_2 = 10 if sv else None
        self.wgamma_2_star = 10 if sv else None
        self.wgamma_3 = 1 if sv else None
        self.wgamma_3_star = 1 if sv else None
        # yearly season
        self.ygamma_1 = 500 if sv else None
        self.ygamma_1_star = -200 if sv else None
        self.ygamma_2 = 100 if sv else None
        self.ygamma_2_star = 100 if sv else None
        self.ygamma_3 = 50 if sv else None
        self.ygamma_3_star = -10 if sv else None
        self.ygamma_4 = -10 if sv else None
        self.ygamma_4_star = 10 if sv else None
        self.ygamma_5 = 10 if sv else None
        self.ygamma_5_star = 10 if sv else None
        self.ygamma_6 = 5 if sv else None
        self.ygamma_6_star = 5 if sv else None
        # variances

        scale = 100
        self.H = 1 * scale if sv else None
        self.Q_mu = 0.01 * scale if sv else None
        self.Q_nu = 0.1 * scale if sv else None
        self.Q_weekly = 0.01 * scale if sv else None
        self.Q_yearly = 0.01 * scale if sv else None

    def set_via_array(self, parameters: ndarray):
        """Set the estimated parameters via an array

        Args:
            parameters (ndarray): The array with parameters you want to set
        """

        # Retrieve the names of each variable name of the class
        attrs = list(vars(self).keys())

        # Set each variable in the class with the value of the given parameter
        for attr, value in zip(attrs, parameters):
            setattr(self, attr, value)

    def get_values(self) -> list[float]:
        """Returns all estimated variables in a list

        Returns:
            list[float]: the list with estimated variables
        """
        # Vars returns a dictionary of all variables in the class with its value, get the values of that dictionary and return in a list
        return list(vars(self).values())
