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

from gcp_rssb.data import PoissonProcess, PoissonProcess2d
from ortho.basis_functions import (
    Basis,
    standard_chebyshev_basis,
    smooth_exponential_basis,
    smooth_exponential_basis_fasshauer,
)

import matplotlib.pyplot as plt
import tikzplotlib
from gcp_rssb.methods.gcp_ose import (
    OrthogonalSeriesCoxProcess,
    GCPOSEHyperparameters,
)
import time

# from gcp_ose_bayesian import BayesianOrthogonalSeriesCoxProcess
# from gcp_rssb.methods.gcp_ose_classifier import loop_hafnian_estimate


@dataclass
class PriorParameters:
    mean: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    nu: torch.Tensor


@dataclass
class DataInformedPriorParameters:
    def __init__(self, nu: torch.Tensor):
        self.alpha = torch.tensor(1.5)
        self.beta = torch.tensor(0.0)
        self.mu = torch.tensor(0.0)
        self.nu = nu


class BayesianOrthogonalSeriesCoxProcess(OrthogonalSeriesCoxProcess):
    """
    Represents the Bayesian interpretation of the Gaussian
    Cox Process method.
    """

    def __init__(
        self,
        gcp_ose_hyperparameters: GCPOSEHyperparameters,
        prior_parameters: PriorParameters,
    ):
        self.hyperparameters = gcp_ose_hyperparameters
        self.prior_parameters = prior_parameters

    def add_data(self, data_points: torch.Tensor, domain: torch.Tensor = None):
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
        if domain is not None:
            self.domain = domain
        start_value = time.perf_counter()
        self.ose_coeffics = self._get_ose_coeffics()

        # get the eigenvalues and mean estimates as posterior mean
        self.posterior_mean_coefficients = (
            self._get_posterior_mean_coefficients()
        )
        self.eigenvalues = self._get_posterior_eigenvalue_estimates()

        self.posterior_mean = self._get_posterior_mean()
        end_value = time.perf_counter()
        print(
            "Time taken to get estimates data: {}".format(
                end_value - start_value
            )
        )

    def _get_posterior_mean_coefficients(self):
        """ """
        print(self.ose_coeffics)
        return self.ose_coeffics / (self.prior_parameters.nu + 1)

    def _get_posterior_eigenvalue_estimates(self):
        """ """
        alpha = self.prior_parameters.alpha
        beta = self.prior_parameters.beta
        nu = self.prior_parameters.nu

        numerator = 2 * beta + (nu / (nu + 1)) * self.ose_coeffics**2
        denominator = 2 * alpha - 1
        eigenvalues = numerator / denominator
        return eigenvalues

    def _get_posterior_mean(self):
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
            self.hyperparameters.basis, self.posterior_mean_coefficients
        )

    def get_posterior_predictive_sample(self):
        """
        Returns a tensor of size (n_points, n_dim) where n_points is a Poisson distributed random variable.
        """
        # step 1: generate a sample intensity function
        intensity_sample = self._get_intensity_sample()
        # breakpoint()

        # step 2: generate a sample of the Poisson process
        dimension = self.domain.shape[0]
        if dimension == 1:
            x_axis = torch.linspace(
                self.domain[0][0] - 0.1, self.domain[0][1] + 0.1, 1000
            )
            bound = torch.max(intensity_sample(x_axis))
            poisson_process = PoissonProcess(
                intensity_sample, self.domain[0][1], bound
            )
        else:
            poisson_process = PoissonProcess2d(
                intensity_sample,
                domain=self.domain,
            )
        poisson_process.simulate()
        return poisson_process.get_data()

    def _get_intensity_sample(self):
        """
        Returns a sample from the posterior distribution of the intensity function.
        """
        mean_coeffics = self.posterior_mean_coefficients
        eigenvalue_std = torch.sqrt(self.eigenvalues)
        random_component = torch.randn_like(self.eigenvalues)
        random_coeffics = eigenvalue_std * random_component
        intensity_sample = HilbertSpaceElement(
            self.hyperparameters.basis, mean_coeffics + random_coeffics
        )
        if torch.any(random_coeffics != random_coeffics):
            breakpoint()
        return intensity_sample


class BayesianOrthogonalSeriesCoxProcessObservationNoise(
    BayesianOrthogonalSeriesCoxProcess
):
    """
    Represents the Bayesian interpretation of the Gaussian
    Cox Process method.
    """

    def __init__(
        self,
        gcp_ose_hyperparameters: GCPOSEHyperparameters,
        prior_parameters: PriorParameters,
    ):
        self.hyperparameters = gcp_ose_hyperparameters
        self.prior_parameters = prior_parameters

    def add_data(self, data_points: torch.Tensor):
        """
        Adds the data to the model.
        """
        self.data_points = data_points
        if domain is not None:
            self.domain = domain

        start_value = time.perf_counter()
        self.ose_coeffics = self._get_ose_coeffics()
        self.ose_square_coeffics = self._get_ose_square_coeffics()

        # get the eigenvalues and mean estimates as posterior mean
        self.posterior_mean_coefficients = (
            self._get_posterior_mean_coefficients()
        )
        end_value = time.perf_counter()
        print(
            "Time taken to get estimates data: {}".format(
                end_value - start_value
            )
        )
        self.eigenvalues = self._get_posterior_eigenvalue_estimates()

        self.posterior_mean = self._get_posterior_mean()

    def _get_posterior_eigenvalue_estimates(self):
        """ """
        eigenvalues = super()._get_posterior_eigenvalue_estimates()
        eigenvalues -= self.ose_square_coeffics
        return eigenvalues

    def _get_ose_square_coeffics(self):
        """
        Calculates the estimates of the variance of the
        orthogonal series estimates given the data points and
        the basis functions.

        According to Kingman's writeup of  Campbell's theorem (p. 26, Kingman 1993),
        the variance of the orthogonal series estimates is given by
                         \sum_{i=1}^n \phi_j^2(x_i)

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

    def get_posterior_predictive_sample(self):
        """
        Returns a tensor of size (n_points, n_dim) where n_points is a Poisson distributed random variable.
        """
        # step 1: generate a sample intensity function
        intensity_sample = self._get_intensity_sample()

        # step 2: generate a sample of the Poisson process
        x_axis = torch.linspace(
            self.domain[0][0] - 0.1, self.domain[0][1] + 0.1, 1000
        )
        bound = torch.max(intensity_sample(x_axis))
        poisson_process = PoissonProcess(
            intensity_sample, self.domain[0][1], bound
        )
        poisson_process.simulate()
        return poisson_process.get_data()

    def _get_intensity_sample(self):
        """
        Returns a sample from the posterior distribution of the intensity function.
        """
        mean_coeffics = self.posterior_mean_coefficients
        eigenvalue_std = torch.sqrt(self.eigenvalues)
        random_component = torch.randn_like(self.eigenvalues)
        random_coeffics = eigenvalue_std * random_component
        intensity_sample = HilbertSpaceElement(
            self.hyperparameters.basis, mean_coeffics + random_coeffics
        )
        if torch.any(random_coeffics != random_coeffics):
            breakpoint()
        return intensity_sample


class DataInformedBayesOSECP(BayesianOrthogonalSeriesCoxProcess):
    """
    The DataInformedBayesOSECP is a Bayesian Orthogonal Series Cox Process
    with observation noise, with data-informed prior.

    To construct this, we actually only have to construct the
    observation-noise-aware Bayesian Orthogonal Series Cox Process, using the
    following values for the first three prior parameters:
        mean: 0
        alpha: 1.5
        beta: \sum_{i=1}^n \phi_j(x_i)

    This is equivalent to setting beta to 0 in the standard
    BayesianOrthogonalSeriesCoxProcess, as then we are subtracting precisely
    the correct amount from the eigenvalue estimates.

    """

    def __init__(
        self,
        hyperparameters: GCPOSEHyperparameters,
        data_informed_prior_parameters: DataInformedPriorParameters,
    ):
        prior_parameters = PriorParameters(
            mean=torch.tensor(0.0),
            alpha=torch.tensor(1.5),
            beta=torch.tensor(0.0),
            nu=data_informed_prior_parameters.nu,
        )
        super().__init__(hyperparameters, prior_parameters)


if __name__ == "__main__":
    plot_intensity = False
    run_classifier = True
    run_standard = False
    run_observation_noise = True

    # intensity and basis function construction
    max_time = 10.0
    alpha_1 = 8.0
    beta_1 = 1.0
    intensity_1 = lambda x: 100 * torch.exp(
        D.Gamma(alpha_1, beta_1).log_prob(x)
    )
    alpha_2 = 3.0
    beta_2 = 1.0
    intensity_2 = lambda x: 100 * torch.exp(
        D.Gamma(alpha_2, beta_2).log_prob(x)
    )
    x = torch.linspace(0.1, max_time, 1000)
    if plot_intensity:
        plt.plot(
            x.cpu().numpy(),
            intensity_1(x).cpu().numpy(),
        )
        plt.plot(
            x.cpu().numpy(),
            intensity_2(x).cpu().numpy(),
        )
        plt.show()
    poisson_process_1 = PoissonProcess(intensity_1, max_time)
    poisson_process_2 = PoissonProcess(intensity_2, max_time)
    poisson_process_1.simulate()
    poisson_process_2.simulate()

    class_1_data = poisson_process_1.get_data()
    class_2_data = poisson_process_2.get_data()

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

    # prior parameters
    prior_mean = torch.tensor(0.0)
    alpha = torch.tensor(4.0)
    beta = torch.tensor(4.0)
    nu = torch.tensor(0.17)
    prior_parameters = PriorParameters(prior_mean, alpha, beta, nu)

    # cox hyper parameters
    hyperparameters = GCPOSEHyperparameters(
        basis=ortho_basis, dimension=dimension
    )
