"""
This file runs experiments with the 2d tree datasets.
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
    dim = 2
    basis_functions = [standard_chebyshev_basis] * 2

    """
    Now we load each of the data sets and run the model.
    """
    # load the data
    dataset_names = [
        "new_zealand.csv",
        "redwood_full.csv",
        "redwoods_california.csv",
        "swedish_pines.csv",
        "white_oak.csv",
    ]
    data_sets = []
    for data_set_name in dataset_names:
        df = pd.read_csv("spatial-2D/{}".format(data_set_name))
        data_set = torch.tensor(df.values)
        data_sets.append(data_set)
        print(torch.max(data_set))

    posterior_means_list = []
    posterior_mean_coeffics_list = []
    eigenvalues_list = []

    for data_set, data_set_name in zip(data_sets, dataset_names):
        # get the minimum/maximum time and use them to set the bounds
        min_time_x = data_set[:, 0].min()
        max_time_x = data_set[:, 0].max()
        max_time_y = data_set[:, 1].max()
        min_time_y = data_set[:, 1].min()

        # 2d plotting stuff
        x_axis = torch.linspace(min_time_x, max_time_x, 1000)
        y_axis = torch.linspace(min_time_y, max_time_y, 1000)
        X, Y = torch.meshgrid(x_axis, y_axis)
        Z = torch.stack((X, Y), dim=2)
        Z = Z.reshape(-1, 2)

        parameters: List = [
            {
                "lower_bound": min_time_x - 0.1,
                "upper_bound": max_time_x + 0.1,
                #                "variance_parameter": 1.0,
            },
            {
                "lower_bound": min_time_y - 0.1,
                "upper_bound": max_time_y + 0.1,
                #                "variance_parameter": 1.0,
            },
        ]
        order = 20
        print("data set shape: {}".format(data_set.shape))
        print("data set name: {}".format(data_set_name))

        chebyshev_basis = Basis(basis_functions, dim, order, parameters)
        gcp_ose_hyperparameters = GCPOSEHyperparameters(
            basis=chebyshev_basis,
            dimension=dim,
        )
        prior_parameters = PriorParameters(
            mean=0.0, alpha=1.5, beta=2.0, nu=0.12
        )
        gcp_ose_model = BayesianOrthogonalSeriesCoxProcessObservationNoise(
            gcp_ose_hyperparameters, prior_parameters
        )
        gcp_ose_model.add_data(data_set)

        # get the various stuff
        eigenvalues = (
            gcp_ose_model._get_posterior_eigenvalue_estimates().cpu().numpy()
        )
        eigenvalues_list.append(eigenvalues)
        posterior_mean_coeffics = (
            gcp_ose_model._get_posterior_mean_coefficients().cpu().numpy()
        )
        posterior_mean_coeffics_list.append(posterior_mean_coeffics)
        posterior_mean = gcp_ose_model._get_posterior_mean()
        posterior_means_list.append(
            posterior_mean
        )  # something bad is happening here.

        # plot posterior mean
        output = posterior_mean(Z).cpu().numpy().reshape(1000, 1000)
        plt.contourf(X.cpu().numpy(), Y.cpu().numpy(), output, cmap="YlOrRd")
        plt.contour(X.cpu().numpy(), Y.cpu().numpy(), output)
        plt.colorbar()
        plt.scatter(data_set.cpu().numpy()[:, 0], data_set.cpu().numpy()[:, 1])
        plt.show()

    # eigenvalues plot
    for eigenvalues in eigenvalues_list:
        plt.plot(eigenvalues)
    plt.show()

    # posterior mean coefficients plot
    for posterior_mean_coeffics in posterior_mean_coeffics_list:
        plt.plot(posterior_mean_coeffics)
    plt.show()
