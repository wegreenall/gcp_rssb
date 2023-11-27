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


class OrthogonalSeriesCoxProcess(Method):
    """
    It does not look clear that the thing is ever a contraction.
    """

    def __init__(
        self,
        gcp_ose_hyperparameters: GCPOSEHyperparameters,
        sigma: torch.Tensor,
    ):
        self.hyperparameters = gcp_ose_hyperparameters
        self.data_points = None
        self.ose_coeffics = None
        self.trained = False
        self.sigma = sigma

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
        self.posterior_mean = self._get_posterior_mean()

        # set up the mapping
        self.mapping = Mapping(
            self.hyperparameters.basis,
            self.data_points,
            self.posterior_mean,
            self.sigma,
        )

    def get_kernel(self, left_points, right_points):
        pass

    def train(self) -> None:
        """
        Trains the model by iterating the mapping until convergence.
        """
        if self.data_points is None:
            raise ValueError(
                "No data points are set. Please add data to the model first."
            )
        if self.trained:
            raise ValueError(
                "The model is already trained. Please reset the model first."
            )

        if not self.trained:
            # set up the initial guess: 1/λ where λ = 1
            previous_guess = torch.ones(self.hyperparameters.basis.order)

            # iterate until convergence
            new_guess = self.mapping(previous_guess)
            while torch.norm(new_guess - previous_guess) > 1e-6:
                previous_guess = new_guess
                new_guess = self.mapping(previous_guess)

            self.eigenvalues = 1 / new_guess
            self.trained = True
        else:
            raise ValueError("The model is already trained.")

    def predict(self, test_points):
        pass

    def evaluate(self, test_points, method):
        pass

    def _get_ose_coeffics(self) -> torch.Tensor:
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
        design_matrix = basis(self.data_points)
        coeffics = torch.sum(design_matrix, dim=0)
        return coeffics

    def _get_posterior_mean(self) -> HilbertSpaceElement:
        """
        Calculates the posterior mean of the intensity function.

        Return shape: [order]
        """
        if self.ose_coeffics is None:
            raise ValueError(
                "The orthogonal series estimate coefficients are not set.\
                Please add data to the model first."
            )
        return HilbertSpaceElement(
            self.hyperparameters.basis, self.ose_coeffics
        )
