import torch
import numpy as np
import matplotlib.pyplot as plt
from data import PoissonProcess


def manual_bound_example():
    """
    Simulation of Poisson Process with intensity
    λ(t) = 4 * [cos(t) + 1] at interval [0, 2π]

    bound: λ* = 8 >= λ(t) for all t
    expected points: int_{0}_{2π} {λ(t) dt} = E[N(0, 2π)] = 8π
    """

    intensity = lambda t: 4 * (1 + torch.cos(t))
    max_time = 2 * np.pi
    bound = 8.

    poisson_process = PoissonProcess(intensity, max_time, bound)
    poisson_process.simulate()
    points = poisson_process.get_data()

    X = torch.linspace(0, max_time, 200)
    Y = intensity(X)
    plt.plot(X, Y, label='true intensity')
    plt.plot(X, torch.ones_like(X) * bound, label='bound for intensity')
    plt.scatter(points, torch.zeros_like(points), s=7., marker='x', label='simulated data')
    plt.title('simulated data, manual bound')
    plt.legend()
    plt.show()

def automatic_bound_example():
    intensity = lambda t: 4 * (1 + torch.cos(t))
    max_time = 2 * np.pi

    poisson_process = PoissonProcess(intensity, max_time)
    poisson_process.simulate()
    points = poisson_process.get_data()
    bound = poisson_process.get_bound()

    X = torch.linspace(0, max_time, 200)
    Y = intensity(X)
    plt.plot(X, Y, label='true intensity')
    plt.plot(X, torch.ones_like(X) * bound, label='bound for intensity')
    plt.scatter(points, torch.zeros_like(points), s=7., marker='x', label='simulated data')
    plt.title('simulated data, automatic bound')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    manual_bound_example()
    automatic_bound_example()
