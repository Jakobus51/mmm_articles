from data_generation.custom_dgp import CustomDGP
from data_generation.fake_marketing import FakeMarketing
from ssm_model.run_model import run_ssm_model_for_range
from ssm_model.analyze_ssm import analyze_ssm_models
from robyn.analyze_robyn_and_true import analyze_robyn_models

def generate_data_sets(start: int, end: int):
    """generate data sets for the given range, also save them locally

    Args:
        start (int): start range
        end (int): end range
    """
    random_states = range(start, end)
    for random_state in random_states:
        # Generate custom marketing data which contains marketing and promo variables
        marketing_data = FakeMarketing(random_state).df
        # Use the marketing data to generate the observations (revenue) and save the data set
        dgp = CustomDGP(random_state, marketing_data, True, False)


if __name__ == "__main__":
    start_state = 0
    end_state = 100

    # Generate data sets
    #generate_data_sets(start_state, end_state)

    # Run the ssm model
    # This finds all the model parameters using the state space model
    run_ssm_model_for_range(start_state, end_state)
    
    #Analyze the robyn model
    #The Robyn model parameters are retrieved by running Robyn in R.
    # All the relevant variables are saved locally and retrieved in this method
    analyze_robyn_models(start_state, end_state)
