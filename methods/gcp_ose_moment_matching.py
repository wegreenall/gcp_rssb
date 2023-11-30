import torch
import torch.distributions as D
from gcp_rssb.method import Method
from gcp_rssb.data import Data

from ortho.basis_functions import Basis
from mercergp.MGP import HilbertSpaceElement
from mercergp.eigenvalue_gen import (
    EigenvalueGenerator,
    SmoothExponentialFasshauer,
)
from dataclasses import dataclass
from typing import Callable

from gcp_rssb.data import PoissonProcess
from ortho.basis_functions import (
    Basis,
    standard_chebyshev_basis,
    smooth_exponential_basis,
    smooth_exponential_basis_fasshauer,
)

import matplotlib.pyplot as plt
import tikzplotlib
from termcolor import colored
from gcp_rssb.methods.gcp_ose import (
    OrthogonalSeriesCoxProcess,
    GCPOSEHyperparameters,
)


class MomentMatchingOrthogonalSeriesCoxProcess(OrthogonalSeriesCoxProcess):
    def __init__(self, gcp_ose_hyperparameters: GCPOSEHyperparameters):
        self.hyperparameters = gcp_ose_hyperparameters
        self.data_points = None
        self.ose_coeffics = None
        self.eigenvalues = None

    def add_data(self, data_points) -> None:
        """
        Adds the data to the model.


        Specifically, it:
            - stores the data on this class
            - calculates the orthogonal series estimate coefficients of the
              intensity function
            - calculates the posterior mean of the intensity function
            - sets up the Mapping, for which the data points are a prerequisite
        """
        self.data_points = data_points
        self.ose_coeffics = self._get_ose_coeffics()
        self.eigenvalues = self._get_ose_square_coeffics()

    def _get_ose_square_coeffics(self) -> torch.Tensor:
        """
        Calculates the orthogonal series estimate given the data points and
        the basis functions.

        Return shape: [order]
        """
        if self.data_points is None:
            raise ValueError(
                "No data points are set. Please add data to the model first."
            )
        basis = self.hyperparameters.basis
        design_matrix = basis(self.data_points) ** 2
        coeffics = torch.sum(design_matrix, dim=0)
        return coeffics

    def get_eigenvalues(self):
        if self.data_points is None:
            raise ValueError(
                "No data points are set. Please add data to the model first."
            )

        # get the eigenvalues
        return self.eigenvalues


if __name__ == "__main__":
    plot_intensity = False
    run_classifier = True

    # intensity and basis function construction
    max_time = 10.0
    alpha = 8.0
    beta = 1.0
    intensity = lambda x: 100 * torch.exp(D.Gamma(alpha, beta).log_prob(x))
    x = torch.linspace(0.1, max_time, 1000)
    if plot_intensity:
        plt.plot(
            x,
            intensity(x),
        )
        plt.show()
    poisson_process = PoissonProcess(intensity, max_time)
    poisson_process.simulate()
    sample_data = poisson_process.get_data()
    basis_functions = standard_chebyshev_basis
    parameters: dict = {
        "lower_bound": 0.0,
        "upper_bound": max_time + 0.1,
        "chebyshev": "second",
    }
    dimension = 1
    order = 6
    ortho_basis = Basis(basis_functions, dimension, order, parameters)
    sigma = torch.tensor(8.0)
    hyperparameters = GCPOSEHyperparameters(
        basis=ortho_basis, dimension=dimension
    )
    gcp_ose = MomentMatchingOrthogonalSeriesCoxProcess(hyperparameters)

    # add the data
    gcp_ose.add_data(sample_data)
    posterior_mean = gcp_ose._get_posterior_mean()
    eigenvalues = gcp_ose.get_eigenvalues()
    plt.plot(x, posterior_mean(x))
    plt.plot(x, intensity(x))
    plt.legend(["Posterior mean estimate", "Intensity"])
    plt.scatter(sample_data, torch.zeros_like(sample_data), marker=".")
    plt.show()

    # generate some random samples
    sample_count = 10
    x_axis = torch.linspace(0.0, max_time, 1000)
    for i in range(sample_count):
        sample_coeffics = D.MultivariateNormal(
            gcp_ose.ose_coeffics, torch.diag(eigenvalues)
        ).sample()
        sample = ortho_basis(x_axis) @ sample_coeffics
        plt.plot(x_axis, sample)

    plt.show()
