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
from typing import Callable, List

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
from gcp_rssb.methods.gcp_ose_moment_matching import (
    MomentMatchingOrthogonalSeriesCoxProcess,
)
from gcp_rssb.methods.gcp_ose_bayesian import PriorParameters
from gcp_rssb.methods.gcp_ose_classifier import loop_hafnian_estimate

torch.set_default_dtype(torch.float64)

# set to cuda
torch.set_default_tensor_type(torch.cuda.DoubleTensor)


class BayesianPointProcessClassifier:
    def __init__(
        self,
        order: int,
        basis: Basis,
        dimension: int,
        prior_parameters: PriorParameters,
    ):
        self.order = order
        self.basis = basis
        self.dimension = dimension
        self.prior_parameters = PriorParameters

        # set up classes
        self.class_count: int = 0
        self.classes_data: List[torch.Tensor] = []
        self.gcps: List[OrthogonalSeriesCoxProcess] = []

    def add_class(
        self,
        data: torch.Tensor,
        gcp_ose_hyperparameters: GCPOSEHyperparameters,
    ):
        """
        Add a new class via a set of data. Initialises a new GCP.
        """
        # add data
        self.classes_data.append(data)
        self.class_count += 1

        # initialise a new GCP
        gcp = OrthogonalSeriesCoxProcess(
            self.order,
            self.basis,
            self.dimension,
            GCPOSEHyperparameters(),
        )
        gcp.add_data(data)
        self.gcps.append(gcp)

    def get_posterior_mean(self):
        """
        Returns the posterior mean of the classifier.
        """
        for i, gcp in enumerate(self.gcps):
            posterior_mean[i, :] = gcp.get_posterior_mean()
        return posterior_mean


if __name__ == "__main__":
    plot_intensity = True
    run_classifier = True

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
    hyperparameters = GCPOSEHyperparameters(
        basis=ortho_basis, dimension=dimension
    )
    gcp_ose_1 = BayesianPointProcessClassifier(hyperparameters)
    gcp_ose_2 = BayesianPointProcessClassifier(hyperparameters)

    # add the data
    gcp_ose_1.add_data(class_1_data)
    gcp_ose_2.add_data(class_2_data)

    order = 6
    basis_functions = standard_chebyshev_basis
    dimension = 1
    ortho_basis = Basis(basis_functions, dimension, order, parameters)

    # model parameters
    eigenvalue_generator = SmoothExponentialFasshauer(order)
    hyperparameters = GCPOSEHyperparameters(
        basis=ortho_basis, dimension=dimension
    )
    gcp_ose_1 = MomentMatchingOrthogonalSeriesCoxProcess(hyperparameters)
    gcp_ose_1.add_data(class_1_data)
    posterior_mean_1 = gcp_ose_1._get_posterior_mean()
    gcp_ose_2 = MomentMatchingOrthogonalSeriesCoxProcess(hyperparameters)
    gcp_ose_2.add_data(class_2_data)
    posterior_mean_2 = gcp_ose_2._get_posterior_mean()
    if plot_intensity:
        plt.plot(x.cpu().numpy(), posterior_mean_1(x).cpu().numpy())
        plt.plot(x.cpu().numpy(), intensity_1(x).cpu().numpy())
        plt.show()
        plt.plot(x.cpu().numpy(), posterior_mean_2(x).cpu().numpy())
        plt.plot(x.cpu().numpy(), intensity_2(x).cpu().numpy())
        plt.show()

    gcp_ose_1.eigenvalues /= torch.max(gcp_ose_1.eigenvalues)
    gcp_ose_2.eigenvalue /= torch.max(gcp_ose_2.eigenvalues)
    kernel_1 = gcp_ose_1.get_kernel(class_1_data, class_1_data)
    kernel_2 = gcp_ose_2.get_kernel(class_2_data, class_2_data)

    # breakpoint()
    plt.plot(gcp_ose_1.eigenvalues.cpu().numpy())
    plt.plot(gcp_ose_2.eigenvalues.cpu().numpy())
    plt.legend(("eigenvalues one", "eigenvalues two"))
    plt.show()
    # plt.imshow(kernel_1.cpu())
    # plt.show()
    # plt.imshow(kernel_2.cpu())
    # plt.show()

    # run the classifier
    fineness = 100
    test_axis = torch.linspace(0.0, max_time, fineness)
    estimators = torch.zeros(fineness)
    estimators_2 = torch.zeros(fineness)
    for i, x in enumerate(test_axis):
        class_data_1_augmented = torch.cat((class_1_data, torch.Tensor([x])))
        class_data_2_augmented = torch.cat((class_2_data, torch.Tensor([x])))
        kernel_1_augmented = gcp_ose_1.get_kernel(
            class_data_1_augmented, class_data_1_augmented
        )
        kernel_2_augmented = gcp_ose_2.get_kernel(
            class_data_2_augmented, class_data_2_augmented
        )

        # build the kernel matrix with this augmented object
        numerator = loop_hafnian_estimate(
            kernel_1_augmented, posterior_mean_1(class_data_1_augmented), 1000
        )
        denominator = loop_hafnian_estimate(
            kernel_1, posterior_mean_1(class_1_data), 1000
        )
        numerator_2 = loop_hafnian_estimate(
            kernel_2_augmented, posterior_mean_2(class_data_2_augmented), 1000
        )
        denominator_2 = loop_hafnian_estimate(
            kernel_2, posterior_mean_2(class_2_data), 1000
        )
        estimator = torch.exp(numerator - denominator)
        estimator_2 = torch.exp(numerator_2 - denominator_2)
        estimators[i] = estimator
        estimators_2[i] = estimator_2

    plt.plot(test_axis.cpu().numpy(), torch.Tensor(estimators).cpu().numpy())
    plt.plot(test_axis.cpu().numpy(), torch.Tensor(estimators_2).cpu().numpy())
    plt.show()

    plt.plot(
        test_axis.cpu().numpy(),
        (estimators / (estimators + estimators_2)).cpu().numpy(),
    )
    plt.plot(
        test_axis.cpu().numpy(),
        (estimators_2 / (estimators + estimators_2)).cpu().numpy(),
    )
    plt.show()
