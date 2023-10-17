from pathlib import Path


class SaveLocation:
    DATASETS_LOCATION = Path("")
    ROBYN_INPUT = Path("/Users/jakob/Documents/ACMetric/MMM_article/robyn_input")
    MODEL_OUTPUT = Path("/Users/jakob/Documents/ACMetric/MMM_article/own_model")
    ROBYN_MODELS = Path("/Users/jakob/Documents/ACMetric/MMM_article/results/models")

    ROBYN_100RUNS = Path("/Users/jakob/Documents/ACMetric/MMM_article/100_runs")
    SSM_100RUNS = Path(
        "/Users/jakob/Documents/ACMetric/MMM_article/100_runs/ssm_100_runs/combined_results.csv"
    )


class RobynModels:
    MODELS = [
        "1_242_1",
        "1_276_5",
        "1_279_5",
        "2_137_4",
        "2_223_3",
        "2_249_2",
        "2_274_6",
        "2_275_2",
        "2_283_3",
        "5_266_3",
    ]


class ParameterNames:
    betas = ["beta_1", "beta_2", "beta_3", "beta_4", "beta_5", "beta_6"]
    promos = ["promo_1", "promo_2"]
    thetas = ["theta_1", "theta_2", "theta_3", "theta_4", "theta_5", "theta_6"]
    phis = ["phi_1", "phi_2", "phi_3", "phi_4", "phi_5", "phi_6"]
    rhos = ["rho_1", "rho_2", "rho_3", "rho_4", "rho_5", "rho_6"]
    wgammas = [
        "wgamma_1",
        "wgamma_1_star",
        "wgamma_2",
        "wgamma_2_star",
        "wgamma_3",
        "wgamma_3_star",
    ]
    ygammas_1 = [
        "ygamma_1",
        "ygamma_1_star",
        "ygamma_2",
        "ygamma_2_star",
        "ygamma_3",
        "ygamma_3_star",
    ]
    ygammas_2 = [
        "ygamma_4",
        "ygamma_4_star",
        "ygamma_5",
        "ygamma_5_star",
        "ygamma_6",
        "ygamma_6_star",
    ]
    vars = ["H", "Q_mu", "Q_nu", "Q_weekly", "Q_yearly"]
