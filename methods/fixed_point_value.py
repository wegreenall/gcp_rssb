"""
In this file we implement functions that display the probability 
that the mapping dependent on sigma is a contraction mapping.
"""
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
    Mapping,
    OrthogonalSeriesCoxProcess,
    GCPOSEHyperparameters,
)
from scipy.special import gammainc


def chi_cdf(m, x):
    return gammainc(m / 2, x / 2)


def eventA(sigma, k, input_vector, input_vector_prime, mapping: Mapping):
    # numerator for term 1
    term_1_numerator = (
        (sigma**2) * k * torch.norm(input_vector - input_vector_prime)
    )

    # denominator for term 1
    WPhiF = torch.norm(mapping.WPhi, p="fro")  # ||WPhi||_F
    PhiPhiDiff = torch.norm(
        torch.linalg.inv(
            mapping.design_matrix
            + sigma**2 * torch.diag(input_vector)  # (Φ'Φ + sigma^2 I)^-1
        )
        - torch.linalg.inv(
            mapping.design_matrix + sigma**2 * torch.diag(input_vector_prime)
        ),
        p="fro",
    )
    PhiF = torch.norm(mapping.basis_matrix, p="fro")  # ||Φ||_F
    term_1_denominator = WPhiF * PhiPhiDiff * PhiF

    # term 1
    term_1 = term_1_numerator / term_1_denominator

    # numerator for term 1
    psi = mapping.posterior_mean(mapping.data_points)
    # plt.scatter(mapping.data_points, psi, marker="x", linewidth=0.4)
    # plt.show()
    term_2_numerator = torch.norm(
        mapping.basis_matrix.t() @ psi  # , p=""
    )  # ||Φψ||_F

    term_2 = term_2_numerator
    # print("term 1", term_1)
    # print("term 2", term_2)

    return term_1  # - term_2


def deterministic_fixed_point(
    sigma, k, input_vector, input_vector_prime, mapping: Mapping
):
    """
    In the deterministic case, the fixed point transition happens when
        ||WΦ||_F ||Φ'Φλ||_F <= k / ||(Φ'Φ + sigma^2 M^-1)^-1||_F ||(Φ'Φ + sigma^2 (M')^-1)||_F

    The latter term is increasing in sigma;
    We assume there is a point at which the left hand side is less then the
    right hand side.

    Returning rhs-lhs, this will increase and cross 0 when sigma is large enough
    that the rhs is bigger than the lhs.
    """
    WPhiF = torch.norm(mapping.WPhi, p="fro")  # ||WPhi||_F
    psi = mapping.posterior_mean(mapping.data_points)  # Ψ = Φλ
    PhiPhiLambda = torch.norm(
        mapping.basis_matrix.t() @ psi  # , p=""
    )  # ||Φ'Φλ||_F

    lhs = WPhiF * PhiPhiLambda
    M_term = torch.linalg.inv(
        mapping.design_matrix
        + sigma**2 * torch.diag(input_vector)  # (Φ'Φ + sigma^2 M^{-1})^-1
    )
    M_prime_term = torch.linalg.inv(
        mapping.design_matrix
        + sigma**2 * torch.diag(input_vector)  # (Φ'Φ + sigma^2 (M')^{-1})^-1
    )

    # denominator
    # rank_term = torch.linalg.matrix_rank(M_term)
    # svd_1 = torch.linalg.svdvals(M_term)
    # svd_2 = torch.linalg.svdvals(M_prime_term)
    # denominator = rank_term * torch.max(svd_1) * torch.max(svd_2)

    denominator = torch.norm(M_term, p="fro") * torch.norm(
        M_prime_term, p="fro"
    )
    rhs = k / denominator
    return rhs - lhs


if __name__ == "__main__":
    plot_intensity = True

    # the data
    alpha = 6.0
    beta = 1.0
    max_time = 10.0
    intensity = lambda x: 100 * torch.exp(D.Gamma(alpha, beta).log_prob(x))
    x = torch.linspace(0.1, max_time, 1000)
    poisson_process = PoissonProcess(intensity, max_time)
    poisson_process.simulate()
    sample_data = poisson_process.get_data()

    # the model
    dimension = 1
    order = 7
    basis_functions = standard_chebyshev_basis
    max_time = 12.0
    parameters: dict = {
        "lower_bound": 0.0,
        "upper_bound": max_time + 0.1,
        "chebyshev": "second",
    }
    ortho_basis = Basis(basis_functions, dimension, order, parameters)
    hyperparameters = GCPOSEHyperparameters(basis=ortho_basis)
    sigma = torch.tensor(1.0)
    gcp_ose = OrthogonalSeriesCoxProcess(hyperparameters, sigma)
    gcp_ose.add_data(sample_data)
    posterior_mean = gcp_ose._get_posterior_mean()
    if plot_intensity:
        plt.plot(
            x,
            intensity(x),
        )
        plt.plot(x, posterior_mean(x))
        plt.show()

    # set up the mapping
    mapping = Mapping(ortho_basis, sample_data, posterior_mean, sigma)

    sigmas = torch.linspace(4.0, 17.0, 100)
    values = []
    deterministic_values = []
    deterministic_values_2 = []
    k = 0.29  # some contraction factor
    k_2 = 0.99
    input_vector = 1.1 * torch.ones(order)  # initalise
    input_vector_prime = torch.ones(order)  # initalise

    input_vector = mapping(input_vector)
    input_vector_prime = torch.ones(order)
    for sigma in sigmas:
        # run "event"
        mapping.sigma = sigma

        # this should be positive...
        event_size = eventA(
            sigma, k, input_vector, input_vector_prime, mapping
        )
        deterministic_event_size = deterministic_fixed_point(
            sigma, k, input_vector, input_vector_prime, mapping
        )
        deterministic_event_size_2 = deterministic_fixed_point(
            sigma, k_2, input_vector, input_vector_prime, mapping
        )
        values.append(chi_cdf(order, event_size))
        deterministic_values.append(deterministic_event_size)
        deterministic_values_2.append(deterministic_event_size_2)

    plt.plot(sigmas, deterministic_values)
    plt.plot(sigmas, deterministic_values_2)

    # plot a line of zeros
    plt.plot(sigmas, torch.zeros_like(sigmas), color="black")
    plt.show()
