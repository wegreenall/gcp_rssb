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


@dataclass
class GCPOSEHyperparameters:
    basis: Basis
    dimension: int
    # eigenvalue_generator: EigenvalueGenerator


@dataclass
class SmoothExponentialFasshauerParameters:
    ard_parameter: torch.Tensor
    precision_parameter: torch.Tensor
    variance_parameter: torch.Tensor


class OrthogonalSeriesCoxProcess(Method):
    """
    Represents our Gaussian Cox process method.
    """

    def __init__(
        self,
        gcp_ose_hyperparameters: GCPOSEHyperparameters,
        sigma: torch.Tensor,
        deterministic: bool = False,
    ):
        self.hyperparameters = gcp_ose_hyperparameters
        self.data_points = None
        self.ose_coeffics = None
        self.trained = False
        self.sigma = sigma
        self.deterministic = deterministic

    def add_data(self, data_points: torch.Tensor) -> None:
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
            self.deterministic,
        )

    def get_kernel(self, left_points, right_points):
        # return self.kernel(left_points, right_points)
        eigenvalues = self.eigenvalues
        basis = self.hyperparameters.basis
        design_matrix = basis(left_points)
        design_matrix_prime = basis(right_points)
        kernel_matrix = torch.einsum(
            "ij,jk,lk->il",
            design_matrix,
            torch.diag(eigenvalues),
            design_matrix_prime,
        )
        return kernel_matrix

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
            while torch.norm(new_guess - previous_guess) > 1e-5:
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


class OrthogonalSeriesCoxProcessParameterised(OrthogonalSeriesCoxProcess):
    def __init__(
        self,
        gcp_ose_hyperparameters: GCPOSEHyperparameters,
        sigma: torch.Tensor,
        eigenvalue_generator: EigenvalueGenerator,
        deterministic: bool = False,
    ):
        super().__init__(gcp_ose_hyperparameters, sigma, deterministic)
        self.eigenvalue_generator = eigenvalue_generator
        self.dimension = self.hyperparameters.basis.dimension

    def train(self):
        """
        In this version, instead of naively directly getting the eigenvalues,
        we pass in the eigenvalues from the eigenvalue generator given the
        parameters; we then solve for the parameters each time.

        This then allows us to maintain the prior structure of the eigenvalues.
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
            # initial parameters?
            previous_parameters = [
                {
                    "ard_parameter": torch.Tensor(
                        [1.0]
                    ),  # not identifiable from prec.
                    "precision_parameter": torch.Tensor([0.7]),
                    "variance_parameter": torch.Tensor([1.0]),
                }
            ] * self.dimension
            previous_guess = 1 / self.eigenvalue_generator(previous_parameters)

            # iterate until convergence
            new_guess = 1 / self.mapping(previous_guess)
            new_parameters = self.eigenvalue_generator.inverse(
                new_guess, previous_parameters
            )
            counter = 0

            # begin the loop
            while torch.norm(new_guess - previous_guess) > 1e-5:
                counter += 1
                if counter % 100 == 0:
                    print("Iteration:", counter)
                print("New iteration...")
                print(torch.norm(new_guess - previous_guess))
                previous_guess = new_guess
                previous_parameters = new_parameters

                # update step
                interim_guess = 1 / self.mapping(previous_guess)
                new_parameters = self.eigenvalue_generator.inverse(
                    interim_guess, previous_parameters
                )
                new_guess = 1 / self.eigenvalue_generator(new_parameters)
                plt.plot(1 / new_guess)

            self.eigenvalues = self.eigenvalue_generator(new_parameters)
            plt.plot(self.eigenvalues.cpu(), color="red")
            plt.show()
            self.trained = True
            self.parameters = new_parameters
        else:
            raise ValueError("The model is already trained.")


class Mapping(object):
    """
    Represents the contraction mapping that we iterate in order to generate the
    eigenvalues. We structure it this way so we have a separate class storing
    pre-computed matrices, etc.
    """

    def __init__(
        self,
        basis: Basis,
        data_points: torch.Tensor,
        posterior_mean: HilbertSpaceElement,
        sigma: torch.Tensor,
        deterministic: bool = False,
    ):
        # save the parameters
        self.basis = basis
        self.data_points = data_points.squeeze()
        self.posterior_mean = posterior_mean
        self.deterministic = deterministic

        # set up the mapping
        self.basis_matrix = self._get_basis_matrix()  # Φ
        self.weight_matrix = self._get_weight_matrix()  # W
        self.design_matrix = self._get_design_matrix()  # Φ'Φ
        self.sigma = sigma  # σ
        self.pseudodata = self._get_pseudodata()  # Y

        # # get composite elements
        self.Wy = self._get_Wy()  # W'Y
        self.WPhi = self._get_WPhi()  # W'Φ
        self.PhiY = self._get_PhiY()  # Φ'Y

    def __call__(self, inverse_eigenvalues: torch.Tensor) -> torch.Tensor:
        """
        Iterates the mapping.

        Specifically, given a vector of inverse eigenvalues, returns a vector
        of inverse eigenvalues.

        Return shape: [order, 1]
        """
        # term 1 : W'Y
        term_1 = self.Wy

        # term 2 : WΦ(Φ'Φ + σΛ^{-1})^{-1}Φ'Y
        internal_term = torch.linalg.inv(
            self.design_matrix
            + self.sigma**2 * torch.diag(inverse_eigenvalues)
        )
        term_2 = self.WPhi @ internal_term @ self.PhiY
        result = term_1 - term_2

        result /= self.sigma**2
        return result

    def _get_basis_matrix(self) -> torch.Tensor:
        """
        Calculates the basis matrix.

        Return shape: [n, order]
        """
        return self.basis(self.data_points)

    def _get_weight_matrix(self) -> torch.Tensor:
        """
        Calculates the weight matrix.

        Return shape: [n, order]
        """
        basis_matrix_normalising = torch.sum(self.basis_matrix, dim=0)
        return self.basis_matrix / basis_matrix_normalising

    def _get_design_matrix(self) -> torch.Tensor:
        """
        Calculates the design matrix Φ'Φ.

        Return shape: [order, order]
        """
        return torch.einsum("ji,jk->ik", self.basis_matrix, self.basis_matrix)

    def _get_sigma(self) -> torch.Tensor:
        """
        Calculates the sigma parameter (the variance or "conditioning"
                                        parameter).

        Return shape: [1]
        """
        return self.sigma

    def _get_pseudodata(self) -> torch.Tensor:
        """
        Calculates the pseudodata.

        Return shape: [n, 1]
        """
        if self.deterministic:
            pseudodata = self.posterior_mean(self.data_points)
        else:
            noise = D.Normal(0, 1).sample((self.data_points.shape[0],))
            pseudodata = (
                self.posterior_mean(self.data_points) + self.sigma * noise
            )
        return pseudodata

    # composite elements
    def _get_Wy(self) -> torch.Tensor:
        """
        Calculates the composite element WY.

        Return shape: [order, 1]
        """
        return torch.matmul(self.weight_matrix.t(), self.pseudodata)

    def _get_WPhi(self) -> torch.Tensor:
        """
        Calculates the composite element WΦ.

        Return shape: [order, order]
        """
        return torch.matmul(self.weight_matrix.t(), self.basis_matrix)

    def _get_PhiY(self) -> torch.Tensor:
        """
        Calculates the composite element ΦY.

        Return shape: [n, 1]
        """
        return torch.matmul(self.basis_matrix.t(), self.pseudodata)


if __name__ == "__main__":
    plot_intensity = False
    initial_example = True
    gcp_ose_standard = True
    gcp_ose_alternative = False
    gcp_ose_standard_deterministic = False
    gcp_ose_alternative_deterministic = False
    gcp_ose_multidim = False

    max_time = 10.0
    if initial_example:
        # present an example
        # torch.manual_seed(1)
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

    # set up the basis

    use_chebyshev = True
    # set up the class
    if use_chebyshev:
        basis_functions = standard_chebyshev_basis
        parameters: dict = {
            "lower_bound": 0.0,
            "upper_bound": max_time + 0.1,
            "chebyshev": "second",
        }
        # parameters: dict = {
        # "ard_parameter": torch.Tensor([1.0]),
        # "precision_parameter": torch.Tensor([1.0]),
        # "variance_parameter": torch.Tensor([1.0]),
        # }
    else:
        basis_functions = smooth_exponential_basis_fasshauer
        parameters: dict = {
            "ard_parameter": torch.Tensor([1.0]),
            "precision_parameter": torch.Tensor([1.0]),
            "variance_parameter": torch.Tensor([1.0]),
        }
    dimension = 1
    order = 6
    parameters: dict = {"lower_bound": 0.0, "upper_bound": max_time + 0.1}
    ortho_basis = Basis(basis_functions, dimension, order, parameters)
    sigma = torch.tensor(8.0)
    if gcp_ose_standard:
        hyperparameters = GCPOSEHyperparameters(
            basis=ortho_basis, dimension=dimension
        )
        gcp_ose = OrthogonalSeriesCoxProcess(hyperparameters, sigma)

        # add the data
        gcp_ose.add_data(sample_data)
        posterior_mean = gcp_ose._get_posterior_mean()
        plt.legend(["ose_coeffics"])
        plt.plot(gcp_ose.ose_coeffics)
        plt.show()
        plt.legend(["ose square coeffics"])
        plt.plot(gcp_ose.ose_square_coeffics)
        plt.show()

        plt.plot(x, posterior_mean(x))
        plt.plot(x, intensity(x))
        plt.legend(["Posterior mean estimate", "Intensity"])
        plt.scatter(sample_data, torch.zeros_like(sample_data), marker=".")
        # tikzplotlib.save(
        # "/home/william/phd/tex_projects/jet_presentation/figures/gaussian_cox_process_example.tex"
        # )
        plt.show()

        gcp_ose.train()
        print(gcp_ose.eigenvalues)
        plt.plot(gcp_ose.eigenvalues)
        plt.show()

    if gcp_ose_alternative:
        eigenvalue_generator = SmoothExponentialFasshauer(order)
        hyperparameters = GCPOSEHyperparameters(
            basis=ortho_basis, dimension=dimension
        )
        gcp_ose = OrthogonalSeriesCoxProcessParameterised(
            hyperparameters, sigma, eigenvalue_generator
        )

        # add the data
        gcp_ose.add_data(sample_data)
        posterior_mean = gcp_ose._get_posterior_mean()
        plt.plot(x, posterior_mean(x))
        plt.plot(x, intensity(x))
        plt.show()

        eigenvalues = gcp_ose.train()
        print("############")
        print("Eigenvalues:")
        print(gcp_ose.eigenvalues)
        print("############")
        print("Parameters:")
        print(gcp_ose.parameters)
        plt.plot(gcp_ose.eigenvalues)
        plt.show()

    if gcp_ose_standard_deterministic:
        hyperparameters = GCPOSEHyperparameters(basis=ortho_basis)
        gcp_ose = OrthogonalSeriesCoxProcess(
            hyperparameters, sigma, deterministic=True
        )

        # add the data
        # breakpoint()
        gcp_ose.add_data(sample_data)
        # breakpoint()
        posterior_mean = gcp_ose._get_posterior_mean()
        plt.plot(x, posterior_mean(x))
        plt.plot(x, intensity(x))
        plt.legend(["Posterior mean estimate", "Intensity"])
        plt.scatter(sample_data, torch.zeros_like(sample_data), marker=".")
        # tikzplotlib.save(
        # "/home/william/phd/tex_projects/jet_presentation/figures/gaussian_cox_process_example.tex"
        # )
        plt.show()

        gcp_ose.train()
        print(gcp_ose.eigenvalues)
        plt.plot(gcp_ose.eigenvalues)
        plt.show()

    if gcp_ose_alternative_deterministic:
        eigenvalue_generator = SmoothExponentialFasshauer(order)
        hyperparameters = GCPOSEHyperparameters(basis=ortho_basis)
        gcp_ose = OrthogonalSeriesCoxProcessParameterised(
            hyperparameters, sigma, eigenvalue_generator, deterministic=True
        )

        # add the data
        gcp_ose.add_data(sample_data)
        posterior_mean = gcp_ose._get_posterior_mean()
        # breakpoint()
        # plt.plot(x, posterior_mean(x))
        # plt.plot(x, intensity(x))
        # plt.show()

        eigenvalues = gcp_ose.train()
        print("############")
        print("Eigenvalues:")
        print(gcp_ose.eigenvalues)
        print("############")
        print("Parameters:")
        print(gcp_ose.parameters)
        plt.plot(gcp_ose.eigenvalues)
        plt.show()

    if gcp_ose_multidim:
        dimension = 2
        mv_mean = torch.Tensor([5.0, 5.0])
        covariance = torch.Tensor([[5.0, 0.0], [0.0, 5.0]])

        def multidim_intensity(x):
            # breakpoint()
            dist = D.MultivariateNormal(mv_mean, covariance)
            return 40000 * torch.exp(dist.log_prob(x))

        # intensity_guess = multidim_intensity(torch.Tensor([5.0, 5.0]))
        intensity_guess = multidim_intensity(torch.ones(2))

        # multidim Poisson process
        poisson_process = PoissonProcess(
            multidim_intensity, max_time, dimension=dimension
        )
        poisson_process.simulate()

        sample_data = poisson_process.get_data()
        basis_functions = [standard_chebyshev_basis] * dimension
        order = 6
        parameters: dict = [
            {
                "lower_bound": 0.0,
                "upper_bound": max_time + 0.1,
                "variance_parameter": 1.0,
            },
            {
                "lower_bound": 0.0,
                "upper_bound": max_time + 0.1,
                "variance_parameter": 1.0,
            },
        ]
        ortho_basis = Basis(basis_functions, dimension, order, parameters)
        eigenvalue_generator = SmoothExponentialFasshauer(
            order, dimension=dimension
        )
        hyperparameters = GCPOSEHyperparameters(basis=ortho_basis)
        gcp_ose = OrthogonalSeriesCoxProcessParameterised(
            hyperparameters, eigenvalue_generator
        )

        gcp_ose.add_data(poisson_process.get_data())
        gcp_ose.train()
