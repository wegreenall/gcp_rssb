from dataclasses import dataclass
import torch
from enum import Enum


@dataclass
class Data:
    """A dataclass for storing data."""

    points: torch.Tensor


class Metric(Enum):
    L2 = 1
    L1 = 2
    TEST_LOG_LIKELIHOOD = 3
