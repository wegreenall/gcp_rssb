import torch
from enum import Enum
from abc import ABCMeta, abstractmethod
from torchmin import minimize_constr
from dataclasses import dataclass


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
    ):
        self._data = None
        self.intensity = intensity
        self.bound = bound
        self.max_time = max_time

    def simulate(self):
        """
        Simulate observations from the IHPP with specified intensity function.
        If no bound is provided i.e bould = 0 (since λ(t) >= 0) then such bound
        is derived automatically via optimization
        """
        if self.bound == 0.0:
            negative_intensity = lambda t: -self.intensity(t)
            result = minimize_constr(
                negative_intensity,
                x0=torch.tensor(1.0),
                bounds=dict(lb=0.0, ub=self.max_time),
            )
            self.bound = -result.fun

        # torch.manual_seed(2) #reproducability

        num_of_points = int(
            torch.distributions.Poisson(
                rate=self.max_time * self.bound
            ).sample()
        )

        # samples to be thinned
        homo_samples, _ = torch.sort(
            torch.distributions.Uniform(0, self.max_time).sample(
                torch.Size([num_of_points])
            )
        )

        inhomo_samples = self._reject(homo_samples)
        self._data = inhomo_samples

    def _reject(self, homo_samples: torch.Tensor) -> torch.Tensor:
        """
        :param homo_samples: Samples from the homogeneous Poisson Process
        :return: samples from the inhomogeneous Poisson Process via thinning
        """
        u = torch.rand(len(homo_samples))
        keep_idxs = torch.where(
            u <= self.intensity(homo_samples) / self.bound, True, False
        )
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
