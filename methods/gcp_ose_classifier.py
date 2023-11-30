from gcp_rssb.data import PoissonProcess
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

torch.set_default_dtype(torch.float64)

# set to cuda
torch.set_default_tensor_type(torch.cuda.DoubleTensor)
# torch.set_default_device("cuda")
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


def loop_hafnian_estimate(
    kernel_matrix: torch.Tensor, mean_function: torch.Tensor, det_count: int
) -> torch.Tensor:
    """
    Calculates the Barvinok estimator for the loop hafnian.
    """
    use_random_diagonal = False
    # first calculate the hat kernel matrix
    hat_kernel_matrix_zero_diag = kernel_matrix - torch.diag(kernel_matrix)
    hat_kernel_matrix = hat_kernel_matrix_zero_diag + torch.diag(mean_function)

    # generate the skew-symmetric matrix
    gaussian_matrix = torch.einsum(
        "ijn->nij",
        (
            D.Normal(0.0, 1.0).sample(
                (kernel_matrix.shape[0], kernel_matrix.shape[0], det_count)
            )
        ),
    )
    mask = torch.eye(kernel_matrix.shape[0]).repeat(det_count, 1, 1).bool()
    upper_triangular_matrix = torch.triu(gaussian_matrix)
    lower_triangular_matrix = torch.transpose(upper_triangular_matrix, 1, 2)
    if use_random_diagonal:
        random_matrix = (
            upper_triangular_matrix - lower_triangular_matrix
        ).masked_fill(mask, 0.0)
        random_diagonal = torch.einsum(
            "ijn->nij",
            (
                D.Normal(0.0, 1.0).sample(
                    (kernel_matrix.shape[0], kernel_matrix.shape[0], det_count)
                )
            )
            ** 2,
        )
        random_matrix = torch.where(mask, random_diagonal, random_matrix)
    else:
        random_matrix = (
            upper_triangular_matrix - lower_triangular_matrix
        ).masked_fill(mask, 1.0)

    estimator = torch.mean(
        torch.logdet(random_matrix * hat_kernel_matrix), dim=0
    )
    estimator = torch.nan_to_num(estimator, nan=0.0)
    return estimator


if __name__ == "__main__":
    print("Starting the program!")
    plot_intensity = False  # flag for plotting the intensity.
    plot_data = False  # flag for plotting the data
    train_eigenvalues = False

    # Class 1 point process
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

    # Plot the data
    if plot_data:
        plt.scatter(
            class_1_data,
            torch.zeros_like(class_1_data),
            label="Class 1",
            color="blue",
            marker="x",
            linewidth=0.3,
        )
        plt.scatter(
            class_2_data,
            torch.zeros_like(class_2_data),
            label="Class 2",
            color="red",
            marker="x",
            linewidth=0.3,
        )
        plt.show()

    # now generate the classification method
    # 1. Get the intensity function estimates for each class

    # basis functions
    order = 6
    basis_functions = standard_chebyshev_basis
    dimension = 1
    sigma = 9.5
    parameters: dict = {"lower_bound": 0.0, "upper_bound": max_time + 0.1}
    ortho_basis = Basis(basis_functions, dimension, order, parameters)

    # model parameters
    eigenvalue_generator = SmoothExponentialFasshauer(order)
    hyperparameters = GCPOSEHyperparameters(
        basis=ortho_basis, dimension=dimension
    )
    gcp_ose_1 = OrthogonalSeriesCoxProcessParameterised(
        hyperparameters, sigma, eigenvalue_generator
    )
    gcp_ose_1.add_data(class_1_data)
    # breakpoint()
    posterior_mean_1 = gcp_ose_1._get_posterior_mean()
    gcp_ose_2 = OrthogonalSeriesCoxProcessParameterised(
        hyperparameters, sigma, eigenvalue_generator
    )
    gcp_ose_2.add_data(class_2_data)
    posterior_mean_2 = gcp_ose_2._get_posterior_mean()

    if plot_intensity:
        plt.plot(x.cpu().numpy(), posterior_mean_1(x).cpu().numpy())
        plt.plot(x.cpu().numpy(), intensity_1(x).cpu().numpy())
        plt.show()
        plt.plot(x.cpu().numpy(), posterior_mean_2(x).cpu().numpy())
        plt.plot(x.cpu().numpy(), intensity_2(x).cpu().numpy())
        plt.show()

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
        print(gcp_ose_2.parameters)
        plt.plot(gcp_ose_2.eigenvalues)
        plt.show()

        result = input("Do you want to save the eigenvalues? (y/n)")
        if result == "y":
            torch.save(gcp_ose_1.eigenvalues, "eigenvalues_class_1.pt")
            torch.save(gcp_ose_2.eigenvalues, "eigenvalues_class_2.pt")
    else:
        # now load the eigenvalues
        gcp_ose_1.eigenvalues = torch.load("eigenvalues_class_1.pt").to("cuda")
        gcp_ose_2.eigenvalues = torch.load("eigenvalues_class_2.pt").to("cuda")

    kernel_1 = gcp_ose_1.get_kernel(class_1_data, class_1_data)
    kernel_2 = gcp_ose_2.get_kernel(class_2_data, class_2_data)
    plt.imshow(kernel_1.cpu())
    plt.show()
    plt.imshow(kernel_2.cpu())
    plt.show()

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

    plt.plot(
        test_axis.cpu().numpy(),
        (estimators / (estimators + estimators_2)).cpu().numpy(),
    )
    plt.plot(
        test_axis.cpu().numpy(),
        (estimators_2 / (estimators + estimators_2)).cpu().numpy(),
    )
    plt.show()
