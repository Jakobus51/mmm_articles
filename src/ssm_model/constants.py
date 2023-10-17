from numpy import random
from pathlib import Path


class RandomStates:
    STATE = 41


class ModelSpecifications:
    TRAIN_SIZE_PERC = 1

    # Marketing stuff
    NUMBER_OF_BETAS = 8
    NUMBER_OF_PROMOS = 2
    NUMBER_OF_WEEKLY_GAMMAS = 6
    NUMBER_OF_YEARLY_GAMMAS = 12
    NUMBER_OF_VARIANCES = 5

    # Season sizes
    WEEKLY_SEASON_SIZE = 7
    YEARLY_SEASON_SIZE = 365

    # starting index in alpha and P
    INDEX_BETA = 2  # due to the local level
    INDEX_PROMO = INDEX_BETA + NUMBER_OF_BETAS
    INDEX_W_GAMMA = INDEX_PROMO + NUMBER_OF_PROMOS
    INDEX_Y_GAMMA = INDEX_W_GAMMA + NUMBER_OF_WEEKLY_GAMMAS

    # 3* beta for theta, rho and phi and also 5 variances
    ESTIMATED_PARAMTERS = (
        3 * NUMBER_OF_BETAS
        + NUMBER_OF_PROMOS
        + NUMBER_OF_WEEKLY_GAMMAS
        + NUMBER_OF_YEARLY_GAMMAS
        + NUMBER_OF_VARIANCES
    )

    # Model dimensions
    NUMBER_OF_SIM_OBSERVATIONS = 1  # p in matrix dimensions
    NUMBER_OF_STATE_PARAMETERS = (
        2
        + NUMBER_OF_BETAS
        + NUMBER_OF_PROMOS
        + NUMBER_OF_WEEKLY_GAMMAS
        + NUMBER_OF_YEARLY_GAMMAS
    )  # m in matrix dimensions (+2 is for the local level)
    NUMBER_OF_STATE_VARIANCES = NUMBER_OF_STATE_PARAMETERS  # r in matrix dimensions


class InitialKalmanFilter:
    VARIANCE = 10**5
    STATE_VECTOR = 0


class Optimization:
    TOLERANCE = 10**-6
    ITERATION_MULTIPLIER = 0.1


class Forecast:
    PROBABILITY_OF_INCULSION = 0.5


class SteadyState:
    THRESHOLD = 0.0001


class DiagnosticTests:
    SIGNIFICANCE = 0.05
    SUBSAMPLE_SIZE = 50
    LAGS = 10
