import numpy as np
from scipy.stats import norm


def get_log_likelihood(wrapped_model, observable_names, n_replicates, experimental_data):
    """
    Creates a log likelihood function based on the experimental data and the model.
    """
    likelihoods = {}

    for observable in observable_names:
        y_meas = experimental_data[[col for col in experimental_data.columns if observable in col]].values
        mean = np.mean(y_meas, axis=1)
        var = np.var(y_meas, axis=1)
        likelihoods[observable] = norm(loc=mean, scale=np.sqrt(var))

    def log_likelihood(parameters):
        sim_result = wrapped_model(parameters)
        sim_data_protein = sim_result[observable_names]
        ll = 0

        for observable in observable_names:
            y_pred = sim_data_protein[observable].values.reshape(-1, 1)
            y_meas = experimental_data[[col for col in experimental_data.columns if observable in col]].values
            cur_var = np.var(y_meas, axis=1).reshape(-1, 1)
            exp_part = np.power(y_meas - y_pred, 2) / (2 * cur_var)
            non_exp_part = 0.5 * np.log(2 * np.pi * cur_var)
            cur_ll = non_exp_part + exp_part
            cur_ll[np.isnan(cur_ll)] = 0
            ll += - np.sum(cur_ll)

        return ll

    return log_likelihood
