from ssm_model.estimated_parameters import EstimatedParameters


class TrueParameters:
    true_parameters: EstimatedParameters  # Values to be estimated

    def __init__(self):
        # Set the true parameters which the model has to estimate
        self.true_parameters = EstimatedParameters(False)

        # Begin level of trend
        self.true_parameters.start_trend = 4000

        # Exogenous
        self.true_parameters.beta_1 = 1000
        self.true_parameters.beta_2 = 800
        self.true_parameters.beta_3 = 1000
        self.true_parameters.beta_4 = 200
        self.true_parameters.beta_5 = 750
        self.true_parameters.beta_6 = 800
        self.true_parameters.beta_7 = 400
        self.true_parameters.beta_8 = 1200

        # Adstock (theta = theta)
        self.true_parameters.theta_1 = 0.1
        self.true_parameters.theta_2 = 0.6
        self.true_parameters.theta_3 = 0.8
        self.true_parameters.theta_4 = 0.2
        self.true_parameters.theta_5 = 0.8
        self.true_parameters.theta_6 = 0.3
        self.true_parameters.theta_7 = 0.4
        self.true_parameters.theta_8 = 0.1

        # Curve shape (recommended to be between 0.5 and 3)
        # phi = alpha
        self.true_parameters.phi_1 = 0.6
        self.true_parameters.phi_2 = 0.5
        self.true_parameters.phi_3 = 1.8
        self.true_parameters.phi_4 = 2.8
        self.true_parameters.phi_5 = 1.8
        self.true_parameters.phi_6 = 0.8
        self.true_parameters.phi_7 = 3
        self.true_parameters.phi_8 = 2.2

        # Inflection point (recommended to be between 0.2 and 1)
        # rho = gamma
        self.true_parameters.rho_1 = 0.2
        self.true_parameters.rho_2 = 0.3
        self.true_parameters.rho_3 = 0.3
        self.true_parameters.rho_4 = 0.6
        self.true_parameters.rho_5 = 0.4
        self.true_parameters.rho_6 = 0.8
        self.true_parameters.rho_7 = 0.2
        self.true_parameters.rho_8 = 0.7

        # Promo
        self.true_parameters.promo_1 = 5400
        self.true_parameters.promo_2 = 3400

        # Weekly seasonality
        self.true_parameters.wgamma_1 = 600
        self.true_parameters.wgamma_1_star = -200
        self.true_parameters.wgamma_2 = 100
        self.true_parameters.wgamma_2_star = 10
        self.true_parameters.wgamma_3 = 4
        self.true_parameters.wgamma_3_star = 3

        # Yearly seasonality
        self.true_parameters.ygamma_1 = 1500
        self.true_parameters.ygamma_1_star = -750
        self.true_parameters.ygamma_2 = 300
        self.true_parameters.ygamma_2_star = 300
        self.true_parameters.ygamma_3 = 50
        self.true_parameters.ygamma_3_star = -10
        self.true_parameters.ygamma_4 = -25
        self.true_parameters.ygamma_4_star = 20
        self.true_parameters.ygamma_5 = 10
        self.true_parameters.ygamma_5_star = 10
        self.true_parameters.ygamma_6 = 2
        self.true_parameters.ygamma_6_star = 5

        # Variances
        scale = 1
        self.true_parameters.H = 1000 * scale
        self.true_parameters.Q_mu = 0.01 * scale
        self.true_parameters.Q_nu = 0.1 * scale
        self.true_parameters.Q_weekly = 0.01 * scale
        self.true_parameters.Q_yearly = 0.01 * scale
