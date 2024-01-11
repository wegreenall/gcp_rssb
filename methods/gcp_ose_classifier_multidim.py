f
om gcp_rssb.data import PoissonProcess
from gcp_rssb.methods.gcp_ose import (
    OrthogonalSeriesCoxProcessParameterised,
    GCPOSEHyperparameters,
)
import torch
import torch.distributions as D
import matplotlib.pyplot as plt
from ortho.basis_functions import Basis, standard_chebyshev_basis
from mercergp.eigenvalue_gen import (
    EigenvalueGenerator,
    SmoothExponentialFasshauer,
)
from gcp_rssb.methods.gcp_ose_classifier import loop_hafnian_estimate

"""
In this script we will present the method by which the OSE classifier is set up.

Note 1:
    Suppose each class has a point process on a space $\mathcal{X}.
    The superposition process is the union of all the per-class point processes.
    u: the units or specimens (i.e., images of lentils)
    x(u): the location of the specimen u; i.e. the feature vector associated with u
    y(x): the class of the specimen u
    y^{-1}(r): the set of all specimens of class r
    
    It is shown by McCullagh and Yang that the probability that is assigned to a given
    class should be:
        prob(y(u') = r | data) = \frac{m^(r)(x U x')}{m^r(x)}

    where m^r(x) is the k-point correlation function of the point process of 
    class r at location x. 
    The k-point correlation function of that process is given by:
        m^r(x) = E[X_1.X_2.X_3...X_k]

    i.e., the k-th moment. The result is that the probability of a given class
    requires calculation of the ratio of these k-th moment objects.

    Note however, that in our case, $Λ(χ)$ is technically a Gaussian random 
    variable, at all locations where there were points. As a result, the 
    k-point correlation function yields Isserlis' theorem, (also known as the 
    Wick probability formula) which states that
     E[X_1.X_2.X_3...X_k] = \sum_{\sigma \in S_k} \prod_{{i,j} \in \sigma} E[X_iX_j]
     where S_k is the set of derangements whose square is the identity.
    as a result. This is the hafnian of the covariance matrix.

    As a result, we can apply the Barvinok estimator of the hafnian approach
    to get a definite probability.
"""
import torch
import matplotlib.pyplot as plt


def generate_points_in_square(num_points):
    # Square dimensions
    square_size = 1.0

    # Grid dimensions
    grid_size = 3
    cell_size = square_size / grid_size

    # Generate random points in the square
    points = torch.rand((num_points, 2)) * square_size

    # Determine the class of each point based on the grid
    row_indices = (points[:, 1] // cell_size).long() % grid_size
    col_indices = (points[:, 0] // cell_size).long() % grid_size
    is_black_square = (row_indices + col_indices) % 2 == 0

    # Assign class labels (class 1 for black, class 2 for white)
    point_classes = torch.where(
        is_black_square, torch.tensor(1), torch.tensor(0)
    )
    points_class_1 = points[point_classes == 0]
    points_class_2 = points[point_classes == 1]
    return points_class_1, points_class_2


def get_checkerboard_data(squares_per_side: int, side_length: float):
    """
    Generates a Poisson process data set where the black squares are one clase,
    and white squares are another class.
    """
    # generate some uniform data on the square
    d = 2
    unif = D.Uniform(0.0, side_length).sample((100, d))
    black_square_mask = torch.zeros_like(unif[:, 0])
    for point in unif:
        # generate the mask for the white squares
        if point[0] > side_length / squares_per_side:
            if point[1] > side_length / squares_per_side:
                black_square_mask = black_square_mask + 1
                white_square_mask = (
                    torch.ones_like(unif[:, 0]) - black_square_mask
                )
    # generate the data
    class_1_data = unif[black_square_mask.bool()]
    class_2_data = unif[white_square_mask.bool()]
    # plt.scatter(
    # class_1_data[:, 0], class_1_data[:, 1], marker="x", color="red"
    # )
    # plt.scatter(
    # class_2_data[:, 0], class_2_data[:, 1], marker="o", color="blue"
    # )
    # plt.show()
    return class_1_data, class_2_data


if __name__ == "__main__":
    print("Starting the program!")
    plot_intensity = False  # flag for plotting the intensity.
    plot_data = False  # flag for plotting the data
    train_eigenvalues = False
    train_estimator_values = False
    # data = get_checkerboard_data(3, 10)
    # data = generate_points_in_square(150)
    # Example usage:
    num_points = 550
    # save the data
    if train_estimator_values:
        class_1_data, class_2_data = generate_points_in_square(num_points)
        torch.save(class_1_data, "class_1_data.pt")
        torch.save(class_2_data, "class_2_data.pt")
    else:
        class_1_data = torch.load("class_1_data.pt")
        class_2_data = torch.load("class_2_data.pt")
    axes = plt.axes()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])
    axes.set_aspect("equal")
    # plt.scatter(
    # class_1_data[:, 0].cpu(),
    # class_1_data[:, 1].cpu(),
    # marker="x",
    # color="red",
    # )
    # plt.scatter(
    # class_2_data[:, 0].cpu(),
    # class_2_data[:, 1].cpu(),
    # marker="o",
    # color="blue",
    # )
    # plt.show()

    # Plotting the points with colors based on class
    if plot_data:
        plt.scatter(
            class_1_data[:, 0].cpu(),
            class_1_data[:, 1].cpu(),
            color="red",
            marker="x",
        )
        plt.scatter(
            class_2_data[:, 0].cpu(),
            class_2_data[:, 1].cpu(),
            color="blue",
            marker="o",
        )
        plt.title("Random Points in a Square with a 3x3 Grid")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.show()

    # now generate the classification method
    # 1. Get the intensity function estimates for each class

    # basis functions
    order = 6
    dimension = 2
    basis_functions = [standard_chebyshev_basis] * 2
    sigma = 2
    max_time = 1.0
    extra_window = 0.05
    parameters: dict = [
        {
            "lower_bound": 0.0 - extra_window,
            "upper_bound": max_time + extra_window,
        },
    ] * 2
    ortho_basis = Basis(basis_functions, dimension, order, parameters)

    # model parameters
    eigenvalue_generator = SmoothExponentialFasshauer(order, dimension)
    hyperparameters = GCPOSEHyperparameters(basis=ortho_basis, dimension=2)

    # class 1
    gcp_ose_1 = OrthogonalSeriesCoxProcessParameterised(
        hyperparameters, sigma, eigenvalue_generator
    )
    gcp_ose_1.add_data(class_1_data)
    posterior_mean_1 = gcp_ose_1._get_posterior_mean()

    # class 2
    gcp_ose_1 = OrthogonalSeriesCoxProcessParameterised(
        hyperparameters, sigma, eigenvalue_generator
    )
    gcp_ose_1.add_data(class_1_data)
    gcp_ose_2 = OrthogonalSeriesCoxProcessParameterised(
        hyperparameters, sigma, eigenvalue_generator
    )
    gcp_ose_2.add_data(class_2_data)
    posterior_mean_2 = gcp_ose_2._get_posterior_mean()

    # set up the eigenvalues
    if train_eigenvalues:
        eigenvalues_1 = gcp_ose_1.train()
        print("############")
        print("Eigenvalues Class 1:")
        print(gcp_ose_1.eigenvalues)
        print("############")
        print("Parameters:")
        # print(gcp_ose_1.parameters)

        plt.plot(gcp_ose_1.eigenvalues)
        plt.show()
        eigenvalues_2 = gcp_ose_2.train()
        print("############")
        print("Eigenvalues Class 2:")
        print(gcp_ose_2.eigenvalues)
        print("############")
        print("Parameters:")
        # print(gcp_ose_2.parameters)
        plt.plot(gcp_ose_2.eigenvalues)
        plt.show()

        result = input("Do you want to save the eigenvalues? (y/n)")
        if result == "y":
            torch.save(gcp_ose_1.eigenvalues, "eigenvalues_class_1.pt")
            torch.save(gcp_ose_2.eigenvalues, "eigenvalues_class_2.pt")
    else:
        # now load the eigenvalues
        eigenvalues_1 = torch.load("eigenvalues_class_1.pt").to("cuda")
        eigenvalues_2 = torch.load("eigenvalues_class_2.pt").to("cuda")
        multidim_eigenvalues_1 = torch.flatten(
            torch.outer(eigenvalues_1, eigenvalues_1)
        )
        multidim_eigenvalues_2 = torch.flatten(
            torch.outer(eigenvalues_2, eigenvalues_2)
        )
        gcp_ose_1.eigenvalues = multidim_eigenvalues_1
        gcp_ose_2.eigenvalues = multidim_eigenvalues_2

    kernel_1 = gcp_ose_1.get_kernel(class_1_data, class_1_data)
    kernel_2 = gcp_ose_2.get_kernel(class_2_data, class_2_data)

    # build the grid
    fineness = 50
    test_axis = torch.linspace(0.0, max_time, fineness)
    test_axes_x, test_axes_y = torch.meshgrid(test_axis, test_axis)
    X = torch.vstack((test_axes_x.ravel(), test_axes_y.ravel())).t()
    if train_estimator_values:
        estimators = torch.zeros(fineness**2)
        estimators_2 = torch.zeros(fineness**2)
        indices = []
        for i, x in enumerate(X):
            print(i)
            class_data_1_augmented = torch.cat((class_1_data, x.unsqueeze(0)))
            class_data_2_augmented = torch.cat((class_2_data, x.unsqueeze(0)))
            kernel_1_augmented = gcp_ose_1.get_kernel(
                class_data_1_augmented, class_data_1_augmented
            )
            kernel_2_augmented = gcp_ose_2.get_kernel(
                class_data_2_augmented, class_data_2_augmented
            )

            # build the kernel matrix with this augmented object
            numerator = loop_hafnian_estimate(
                kernel_1_augmented,
                posterior_mean_1(class_data_1_augmented),
                1000,
            )
            denominator = loop_hafnian_estimate(
                kernel_1, posterior_mean_1(class_1_data), 1000
            )
            numerator_2 = loop_hafnian_estimate(
                kernel_2_augmented,
                posterior_mean_2(class_data_2_augmented),
                1000,
            )
            denominator_2 = loop_hafnian_estimate(
                kernel_2, posterior_mean_2(class_2_data), 1000
            )
            estimator = torch.exp(numerator - denominator)
            estimator_2 = torch.exp(numerator_2 - denominator_2)
            indices.append(x)
            estimators[i] = estimator
            estimators_2[i] = estimator_2
        # save the calculated estimators data
        torch.save(estimators, "estimators.pt")
        torch.save(estimators_2, "estimators_2.pt")
    else:
        estimators = torch.load("estimators.pt")
        estimators_2 = torch.load("estimators_2.pt")

    # plt.plot(test_axis, estimators / (estimators + estimators_2))
    # plt.plot(test_axis, estimators_2 / (estimators + estimators_2))
    denominator = estimators + estimators_2
    tensor_estimators = (
        torch.Tensor(estimators).cpu() / torch.Tensor(denominator).cpu()
    )
    tensor_estimators_2 = (
        torch.Tensor(estimators_2).cpu() / torch.Tensor(denominator).cpu()
    )
    # plt.matshow(tensor_estimators.reshape(fineness, fineness))
    plt.matshow(tensor_estimators.reshape(fineness, fineness))
    scaler = 50
    plt.scatter(
        scaler - scaler * class_1_data[:, 0].cpu(),
        scaler - scaler * class_1_data[:, 1].cpu(),
        marker="x",
        color="red",
    )
    plt.scatter(
        scaler - scaler * class_2_data[:, 0].cpu(),
        scaler - scaler * class_2_data[:, 1].cpu(),
        marker="o",
        color="blue",
    )
    plt.colorbar()
    plt.show()
    plt.matshow(tensor_estimators_2.reshape(fineness, fineness))
    plt.colorbar()
    plt.show()
