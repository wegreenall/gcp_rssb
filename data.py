import torch
import torch.distributions as D
from enum import Enum
from abc import ABCMeta, abstractmethod
from torchmin import minimize_constr, minimize
from dataclasses import dataclass
import math
import matplotlib.pyplot as plt


class PoissonProcess2d:
    """
    Class simulating observations from IHPP with
    intensity function λ(t), 0 < t <= max_time
    """

    def __init__(
        self,
        intensity: callable,
        bound: torch.float = 0.0,
        dimension: int = 2,
        domain: torch.Tensor = None,
    ):
        self._data = None
        self.intensity = intensity
        self.bound = bound
        self.dimension = dimension
        self.domain = domain

    def simulate(self) -> None:
        """
        Simulate observations from the IHPP with specified intensity function.
        If no bound is provided i.e bould = 0 (since λ(t) >= 0) then such bound
        is derived automatically via optimization
        """
        # calculate maximiser
        init_point = torch.ones((1, self.dimension))
        bound_top = self.domain[:, 1].unsqueeze(0)
        bound_bottom = self.domain[:, 0].unsqueeze(0)
        if self.bound == 0.0:
            negative_intensity = lambda t: -self.intensity(t)
            result = minimize_constr(
                negative_intensity,
                x0=init_point,
                bounds=dict(lb=bound_bottom, ub=bound_top),
            )
            self.bound = -result.fun

        # get sample point count
        rate_volume = (
            torch.prod(self.domain[:, 1] - self.domain[:, 0]) * self.bound
        )  # this should be right...
        poisson_dist = torch.distributions.Poisson(rate=rate_volume)
        num_of_points = int(poisson_dist.sample())

        if num_of_points == 0:
            print("No points generated!")
            print("rate volume:", rate_volume)
            # print("self.max_time:", self.max_time)

        # set up the ranges for the dimensions
        # generate the homogeneous samples, ready for thinning
        homo_samples = (
            torch.distributions.Uniform(bound_bottom, bound_top)
            .sample(torch.Size([num_of_points]))
            .squeeze()
        )

        # thin the homogeneous samples to get the inhomogeneous values
        inhomo_samples = self._reject(homo_samples)
        self._data = inhomo_samples

    def _reject(self, homo_samples: torch.Tensor) -> torch.Tensor:
        """
        :param homo_samples: Samples from the homogeneous Poisson Process

        :return: samples from the inhomogeneous Poisson Process via thinning
        """
        u = torch.rand(len(homo_samples))
        # print("u.shape:", u.shape)
        # print("homo_samples.shape", homo_samples.shape)
        try:
            values = self.intensity(homo_samples) / self.bound
        except RuntimeError as e:
            print("Watch out!")
            print(e)
        keep_idxs = torch.where(u <= values, True, False)
        if len(keep_idxs) == 0:
            raise ValueError("No values collected in generated sample!")

        return homo_samples[keep_idxs]

    def get_data(self):
        return self._data

    def get_bound(self):
        return self.bound


@dataclass
class Data:
    """A dataclass for storing data."""

    points: torch.Tensor


class Metric(ABCMeta):
    @abstractmethod
    def calculate(
        self, predicted: torch.Tensor, actual: torch.Tensor
    ) -> torch.Tensor:
        pass


class PoissonProcess:
    """
    Class simulating observations from IHPP with
    intensity function λ(t), 0 < t <= max_time
    """

    def __init__(
        self,
        intensity: callable,
        max_time: torch.float,
        bound: torch.float = 0.0,
        dimension: int = 1,
    ):
        self._data = None
        self.intensity = intensity
        self.bound = bound
        self.max_time = max_time
        self.dimension = dimension

    def simulate(self) -> None:
        """
        Simulate observations from the IHPP with specified intensity function.
        If no bound is provided i.e bould = 0 (since λ(t) >= 0) then such bound
        is derived automatically via optimization
        """
        # calculate maximiser
        init_point = torch.ones((self.dimension,))
        bound_top = torch.Tensor([self.max_time] * self.dimension)
        bound_bottom = torch.Tensor([0.0] * self.dimension)
        if self.bound == 0.0:
            negative_intensity = lambda t: -self.intensity(t)
            result = minimize_constr(
                negative_intensity,
                x0=init_point,
                bounds=dict(lb=bound_bottom, ub=bound_top),
            )
            self.bound = -result.fun
        # else:
        # print("bound:", self.bound)

        # get sample point count
        rate_volume = (self.max_time * self.bound) ** self.dimension
        poisson_dist = torch.distributions.Poisson(rate=rate_volume)
        num_of_points = int(poisson_dist.sample())

        if num_of_points == 0:
            print("No points generated!")
            print("self.max_time:", self.max_time)

        # set up the ranges for the dimensions
        # generate the homogeneous samples, ready for thinning
        homo_samples = (
            torch.distributions.Uniform(bound_bottom, bound_top)
            .sample(torch.Size([num_of_points]))
            .squeeze()
        )

        # thin the homogeneous samples to get the inhomogeneous values
        inhomo_samples = self._reject(homo_samples)
        self._data = inhomo_samples

    def _reject(self, homo_samples: torch.Tensor) -> torch.Tensor:
        """
        :param homo_samples: Samples from the homogeneous Poisson Process

        :return: samples from the inhomogeneous Poisson Process via thinning
        """
        u = torch.rand(len(homo_samples))
        # print("u.shape:", u.shape)
        # print("homo_samples.shape", homo_samples.shape)
        try:
            values = self.intensity(homo_samples) / self.bound
        except RuntimeError as e:
            print("Watch out!")
            print(e)
        keep_idxs = torch.where(u <= values, True, False)
        if len(keep_idxs) == 0:
            raise ValueError("No values collected in generated sample!")

        return homo_samples[keep_idxs]

    def get_data(self):
        return self._data

    def get_bound(self):
        return self.bound


@dataclass
class Data:
    """A dataclass for storing data."""

    points: torch.Tensor


class Metric(ABCMeta):
    @abstractmethod
    def calculate(
        self, predicted: torch.Tensor, actual: torch.Tensor
    ) -> torch.Tensor:
        pass


if __name__ == "__main__":
    bounds = torch.Tensor([[0.0, 1.0], [2.0, 3.0]])
    sample_size = 1000
    unif = D.Uniform(bounds[:, 0], bounds[:, 1])
    sample = unif.sample((sample_size,))
    # plt.scatter(sample[:, 0], sample[:, 1])
    # plt.show()

    # test the poisson process stuff
    def intensity(input_points: torch.Tensor):
        return 100 * torch.exp(
            D.MultivariateNormal(
                torch.Tensor([0.0, 0.0]),
                torch.Tensor([[1.0, 0.9], [0.9, 1.0]]),
            ).log_prob(input_points)
        )

    bound = 1000 / math.sqrt(math.pi * 2)
    dimension = 2
    domain = torch.Tensor([[-1, 1], [-1, 1]])
    pp = PoissonProcess2d(intensity, bound, dimension, domain)
    pp.simulate()
    data = pp.get_data()
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()
