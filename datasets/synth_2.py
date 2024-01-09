"""
This file runs experiments with the synthetic data set 2.
"""
import pandas as pd
import torch
import numpy as np

from gcp_rssb.methods.gcp_ose import GCPOSEHyperparameters
from gcp_rssb.methods.gcp_ose_bayesian import (
    PriorParameters,
    BayesianOrthogonalSeriesCoxProcessObservationNoise,
)

from ortho.basis_functions import Basis, standard_chebyshev_basis
from typing import List

import matplotlib.pyplot as plt

if __name__ == "__main__":
    """
    This first section is boilerplate code to set up the model to accept data
    and be used.
    """
    # set up the model
    dim = 1
    basis_functions = [standard_chebyshev_basis]
    min_time = -0.1
    max_time = 5.1
    parameters: List = [
        {
            "lower_bound": min_time - 0.1,
            "upper_bound": max_time + 0.1,
            "variance_parameter": 1.0,
        },
    ]
    order = 10
    chebyshev_basis = Basis(basis_functions, dim, order, parameters)
    gcp_ose_hyperparameters = GCPOSEHyperparameters(
        basis=chebyshev_basis,
        dimension=dim,
    )
    prior_parameters = PriorParameters(mean=0.0, alpha=1.5, beta=2.0, nu=0.00)
    gcp_ose_model = BayesianOrthogonalSeriesCoxProcessObservationNoise(
        gcp_ose_hyperparameters, prior_parameters
    )

    """
    Now we load each of the data sets and run the model.
    """
    # load the data
    synth_2_data_sets = []
    for i in range(10):
        df = pd.read_csv("synth2/observation{}.csv".format(i))
        data_set = torch.tensor(df.values).squeeze()
        synth_2_data_sets.append(data_set)
        print(torch.max(data_set))

    posterior_means_list = []
    posterior_mean_coeffics_list = []
    eigenvalues_list = []
    for data_set in synth_2_data_sets:
        gcp_ose_model.add_data(data_set)

        # get the various stuff
        eigenvalues = (
            gcp_ose_model._get_posterior_eigenvalue_estimates().cpu().numpy()
        )
        eigenvalues_list.append(eigenvalues)
        # plt.plot(eigenvalues)
        posterior_mean_coeffics = (
            gcp_ose_model._get_posterior_mean_coefficients().cpu().numpy()
        )
        posterior_mean_coeffics_list.append(posterior_mean_coeffics)
        posterior_mean = gcp_ose_model._get_posterior_mean()
        posterior_means_list.append(posterior_mean)

    x_axis = torch.linspace(0, max_time, 1000)
    # eigenvalues plot
    for eigenvalues in eigenvalues_list:
        plt.plot(eigenvalues)
    plt.show()

    # posterior mean coefficients plot
    for posterior_mean_coeffics in posterior_mean_coeffics_list:
        plt.plot(posterior_mean_coeffics)
    plt.show()

    for posterior_mean in posterior_means_list:
        plt.plot(posterior_mean(x_axis).cpu().numpy())
    plt.show()
