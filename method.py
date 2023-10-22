import abc
from abc import ABC, abstractmethod
from data import Data, Metric
from typing import Callable

from dataclasses import dataclass

import torch


@dataclass
class MethodHyperparameters:
    pass


class BishopHyperparameters(MethodHyperparameters):
    function: Callable[[torch.Tensor, torch.Tensor, dict], torch.Tensor]
    parameters: dict


class WilliamHyperparameters(MethodHyperparameters):
    eigenvalues: Callable
    basis_function: Callable


class Method(ABC):
    """
    Interface representing method in point process estimation.
    """

    def __init__(self):
        pass

    def add_data(self, data_points: Data):
        self.data_points = data_points

    @abstractmethod
    def get_kernel(
        self, left_points: torch.Tensor, right_points: torch.Tensor
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def train(self):
        """
        Given the data points on self.data_points,
        estimate the parameters of the model; i.e. run the training phase.

        After running this method, we assume that the model is ready to produce
        predictions and evaluations with the corresponding methods.
        """
        pass

    @abstractmethod
    def predict(self, test_points: torch.Tensor) -> torch.Tensor:
        """
        Produce a prediction of the intensity(?) at the test points.
        """
        pass

    @abstractmethod
    def evaluate(
        self, test_points: torch.Tensor, metric: Metric
    ) -> torch.Tensor:
        pass


class Bishop(Method):
    """
    Implementation of Bishop(2018) method.
    """

    def __init__(self, hyperparameters: BishopHyperparameters):
        self.hyperparameters = hyperparameters
        pass

    def get_kernel(self, left_points, right_points) -> torch.Tensor:
        return self.hyperparameters.function(
            left_points, right_points, self.hyperparameters.parameters
        )
        pass

    def train(self):
        pass

    def add_data(self, data_points: Data):
        super().add_data(data_points)

    def predict(self, test_points: torch.Tensor):
        pass

    def estimate(self):
        pass

    def evaluate(self):
        pass


if __name__ == "__main__":
    bishop = Bishop()
    bishop.add_data(Data(torch.Tensor([1, 2, 3])))
    bishop.evaluate(torch.Tensor([1, 2, 3]))
    bishop.estimate()  # run the training
    bishop.predict(torch.Tensor([1, 2, 3]))
    pass
