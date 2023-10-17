from dataclasses import dataclass
import torch
from enum import Enum
from abc import ABCMeta, abstractmethod


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
