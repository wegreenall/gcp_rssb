from gcp_rssb.data import PoissonProcess
from gcp_rssb.methods.gcp_ose import (
    OrthogonalSeriesCoxProcessParameterised,
    GCPOSEHyperparameters,
)
from gcp_rssb.methods.gcp_ose_bayesian import (
    PriorParameters,
    DataInformedPriorParameters,
    BayesianOrthogonalSeriesCoxProcess,
    BayesianOrthogonalSeriesCoxProcessObservationNoise,
    DataInformedBayesOSECP,
)
import torch
import torch.distributions as D
import matplotlib.pyplot as plt
from ortho.basis_functions import Basis, standard_chebyshev_basis
from mercergp.eigenvalue_gen import (
    EigenvalueGenerator,
    SmoothExponentialFasshauer,
)
from typing import Union, List

torch.set_default_dtype(torch.float64)

# set to cuda
# torch.set_default_tensor_type(torch.cuda.DoubleTensor)
# torch.set_default_device("cuda")
from termcolor import colored


class GCPClassifier:
    """
    GCP Classifier: represents the classifier based in the Gaussian Cox process
    method we have developed, and using the Barvinok estimator for the loop
    hafnian to get the k-point correlation function.

    Suppose each class in the data is represented as a point process on a space
    $\mathcal{X}. The superposition process is the union of all the per-class
    point processes. u: the units or specimens (i.e., images of lentils) x(u):
        the location of the specimen u; i.e. the feature vector associated with
        u y(x): the class of the specimen u y^{-1}(r): the set of all specimens
        of class r

    It is shown by McCullagh and Yang that the probability that is assigned to
    a given class should be: prob(y(u') = r | data) = \frac{m^(r)(x U
                                                                  x')}{m^r(x)}

    where m^r(x) is the k-point correlation function of the point process of
    class r at location x. The k-point correlation function of that process is
    given by: m^r(x) = E[X_1.X_2.X_3...X_k]

    i.e., the k-th moment. The result is that the probability of a given class
    requires calculation of the ratio of these k-th moment objects.

    Note however, that in our case, $Λ(χ)$ is technically a Gaussian random
    variable, at all locations where there were points. As a result, the
    k-point correlation function yields Isserlis' theorem, (also known as the
                                                            Wick probability
                                                            formula) which
    states that E[X_1.X_2.X_3...X_k] = \sum_{\sigma \in S_k} \prod_{{i,j} \in
                                                                    \sigma}
    E[X_iX_j] where S_k is the set of derangements whose square is the
    identity. as a result. This is the hafnian of the covariance matrix.

    As a result, we can apply the Barvinok estimator of the hafnian approach to
    get a definite probability.

    The classifier therefore is constructed by building a Gaussian Cox process
    model for each of the classes, and then using the Barvinok estimator to
    approximate its k-point correlation function.
    """

    def __init__(
        self,
        class_count: int,
        parameters: Union[List[PriorParameters], PriorParameters],
        hyperparameters: Union[
            List[GCPOSEHyperparameters], GCPOSEHyperparameters
        ],
        data_informed: bool = False,
        stabilising_epsilon=0.005,
    ):
        self.stabilising_epsilon = stabilising_epsilon
        self.class_count = class_count
        self.data_informed = data_informed
        if (
            isinstance(parameters, DataInformedPriorParameters)
            and not data_informed
        ):
            raise ValueError(
                "You passed a DataInformedPriorParameters object as the\
                    parameter, but didn't set data_informed to True."
            )
        self.cox_processes = self._parse_cox_processes(
            parameters, hyperparameters
        )

    def _parse_cox_processes(
        self,
        parameter: Union[List[PriorParameters], PriorParameters],
        hyperparameters: Union[
            List[GCPOSEHyperparameters], GCPOSEHyperparameters
        ],
    ) -> List[BayesianOrthogonalSeriesCoxProcess]:
        """
        Captures the cases for the different combinations of inputs and
        parses them to create appropriate Cox processes.
        """
        if self.data_informed:
            cox_process_type = DataInformedBayesOSECP
        else:
            cox_process_type = BayesianOrthogonalSeriesCoxProcess

        case_1 = (
            isinstance(parameter, PriorParameters)
            or isinstance(parameter, DataInformedPriorParameters)
        ) and isinstance(hyperparameters, GCPOSEHyperparameters)
        case_2 = isinstance(parameter, list) and isinstance(
            hyperparameters, list
        )
        if case_1:  # single parameter and hyperparameter set
            return [
                cox_process_type(hyperparameters, parameter)
                for _ in range(self.class_count)
            ]
        elif case_2:  # a list of different sets, one set for each process
            return [
                cox_process_type(hyperparameter_set, parameter_set)
                for hyperparameter_set, parameter_set in zip(
                    hyperparameters, parameter
                )
            ]
        else:
            raise ValueError(
                "Please pass either a single parameter and hyperparameter set,\
                or a lists of parameter and hyperparameter sets."
            )

    def add_data(self, data: torch.Tensor, class_index: int):
        """
        Adds data to the appropriate Cox process.
        """
        self.cox_processes[class_index].add_data(data)

    def predict_point(self, test_points: torch.Tensor) -> torch.Tensor:
        """
        Outputs class probabilities for a set of test points.

        Input shape: (N, d)
        Output shape: (N, C)
        """
        log_numerators = torch.zeros(test_points.shape[0], self.class_count)
        log_denominators = torch.zeros(1, self.class_count)

        #        print("About the start the denominators loop inside predict_point")
        # denominators
        for i, cox_process in enumerate(self.cox_processes):
            data_points = cox_process.data_points
            log_denominators[:, i] = self._get_estimators(
                data_points, cox_process
            )
        log_denominators_repeated = log_denominators.repeat(
            test_points.shape[0], 1
        )

        # print("About the start the loop inside predict_point")
        # numerators
        for c, cox_process in enumerate(self.cox_processes):
            data_points = cox_process.data_points
            for i, test_point in enumerate(test_points):
                augmented_data_points = torch.cat(
                    (data_points, test_point.unsqueeze(0))
                )
                log_numerators[i, c] = self._get_estimators(
                    augmented_data_points, cox_process
                )

        # produce the predictions
        raw_predictions = torch.exp(log_numerators - log_denominators_repeated)
        normaliser = (
            torch.sum(raw_predictions, dim=1, keepdim=True)
            + self.stabilising_epsilon
        )
        normalised_predictions = raw_predictions / normaliser
        return normalised_predictions

    def _get_estimators(
        self,
        points: torch.Tensor,
        cox_process: BayesianOrthogonalSeriesCoxProcess,
    ) -> torch.Tensor:
        """
        Returns the log determinant estimators for the given points;
        to this function we pass either the data points augmented with test
        points (numerators) or just the data points (denominators)
        """
        kernel_matrix = cox_process.get_kernel(points, points)
        mean_function = cox_process._get_posterior_mean()
        print(
            "About to get the log_estimators using loop_hafnian_estimate on line 202"
        )
        log_estimator = loop_hafnian_estimate(
            kernel_matrix, mean_function(points), 2000
        )
        print(colored("Finished getting the log_estimators once", "blue"))
        torch.cuda.empty_cache()
        return log_estimator


def loop_hafnian_estimate(
    kernel_matrix: torch.Tensor, mean_function: torch.Tensor, det_count: int
) -> torch.Tensor:
    """
    Calculates the log of the loop hafnian estimator for the k-point correlation
    function of the Gaussian Cox process.

    First, we calculate the augmented kernel matrix, which has the
    mean function at the data (and test point) along its diagonal.
    """
    # first calculate the hat kernel matrix: K + diag(m(x)) - diag(K)
    # Note that here torch.diag(.) does different things in each line
    hat_kernel_matrix_zero_diag = kernel_matrix - torch.diag(kernel_matrix)
    hat_kernel_matrix = hat_kernel_matrix_zero_diag + torch.diag(mean_function)

    # generate the skew-symmetric matrices - these will be the random
    # sample for the Monte Carlo determinant estimator
    gaussian_matrix = torch.einsum(
        "ijn->nij",
        (
            D.Normal(0.0, 1.0).sample(
                (kernel_matrix.shape[0], kernel_matrix.shape[0], det_count)
            )
        ),
    )

    # generate a mask to fill in the diagonals with 1s
    mask = torch.eye(kernel_matrix.shape[0]).repeat(det_count, 1, 1).bool()
    upper_triangular_matrix = torch.triu(gaussian_matrix)
    lower_triangular_matrix = torch.transpose(upper_triangular_matrix, 1, 2)
    random_matrix = (
        upper_triangular_matrix - lower_triangular_matrix
    ).masked_fill(mask, 1.0)

    # construct the final estimator as the mean of the log determinants of the
    # generate random matrices
    determinables = random_matrix * hat_kernel_matrix
    # print(colored("About to move to gpu!", "blue"))
    cuda_determinables = determinables.cuda()
    logdets = torch.logdet(cuda_determinables).to("cpu")
    # print(colored("Just got our stuff back from the GPU!", "red"))
    torch.cuda.empty_cache()
    estimator = torch.mean(logdets, dim=0)
    estimator = torch.nan_to_num(estimator, nan=0.0)
    return estimator


def loop_hafnian_WSM(
    design_matrix: torch.Tensor,
    kernel_matrix: torch.Tensor,
    eigenvalues: torch.Tensor,
    mean_function: torch.Tensor,
    det_count: int,
) -> torch.Tensor:
    # step one: check that the loop hafnian thing works correctly.
    # first calculate the hat kernel matrix: K + diag(m(x)) - diag(K)
    # Note that here torch.diag(.) does different things in each line
    hat_kernel_matrix_zero_diag = kernel_matrix - torch.diag(kernel_matrix)
    hat_kernel_matrix = hat_kernel_matrix_zero_diag + torch.diag(mean_function)
    """
    In the WSM determinant formula, the result is:

        det(A + ΦWΦ') = det(A) * det(W) * det(W^{-1} + Φ'A^{-1}Φ)
    """
    K = kernel_matrix  # ΦWΦ'
    A = torch.diag(mean_function) - torch.diag(kernel_matrix)  # A
    W = torch.diag(eigenvalues)  # W
    # print(K.shape)
    # print(A.shape)
    # print(W.shape)
    LHS_matrix = A + K
    LHS = torch.linalg.det(LHS_matrix)
    RHS_matrix = (
        torch.inverse(W) + design_matrix.t() @ torch.inverse(A) @ design_matrix
    )
    RHS = (
        torch.linalg.det(A)
        * torch.linalg.det(W)
        * torch.linalg.det(RHS_matrix)
    )

    # generate the skew-symmetric matrices - these will be the random
    # sample for the Monte Carlo determinant estimator
    gaussian_matrix_LHS = torch.einsum(
        "ijn->nij",
        (
            D.Normal(0.0, 1.0).sample(
                (LHS_matrix.shape[0], LHS_matrix.shape[0], det_count)
            )
        ),
    )
    gaussian_matrix_RHS = torch.einsum(
        "ijn->nij",
        (
            D.Normal(0.0, 1.0).sample(
                (RHS_matrix.shape[0], RHS_matrix.shape[0], det_count)
            )
        ),
    )

    # generate a mask to fill in the diagonals with 1s # LHS
    mask_LHS = torch.eye(LHS_matrix.shape[0]).repeat(det_count, 1, 1).bool()
    upper_triangular_matrix_LHS = torch.triu(gaussian_matrix_LHS)
    lower_triangular_matrix_LHS = torch.transpose(
        upper_triangular_matrix_LHS, 1, 2
    )

    # generate a mask to fill in the diagonals with 1s # LHS
    mask_RHS = torch.eye(RHS_matrix.shape[0]).repeat(det_count, 1, 1).bool()
    upper_triangular_matrix_RHS = torch.triu(gaussian_matrix_RHS)
    lower_triangular_matrix_RHS = torch.transpose(
        upper_triangular_matrix_RHS, 1, 2
    )

    #
    random_matrix_LHS = (
        upper_triangular_matrix_LHS - lower_triangular_matrix_LHS
    ).masked_fill(mask_LHS, 1.0)
    random_matrix_RHS = (
        upper_triangular_matrix_RHS - lower_triangular_matrix_RHS
    ).masked_fill(mask_RHS, 1.0)

    # construct the final estimator as the mean of the log determinants of the
    # generate random matrices
    determinables_LHS = random_matrix_LHS * LHS_matrix
    determinables_RHS = random_matrix_RHS * RHS_matrix
    logdets_LHS = torch.logdet(determinables_LHS)
    logdets_RHS = (
        torch.logdet(determinables_RHS) + torch.logdet(A) + torch.logdet(W)
    )
    print("LHS:")
    breakpoint()
    print(torch.mean(logdets_LHS, dim=0))
    print("RHS:")
    print(torch.mean(logdets_RHS, dim=0))
    breakpoint()


if __name__ == "__main__":

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

    print("Starting the program!")
    plot_intensity = False  # flag for plotting the intensity.
    plot_data = False  # flag for plotting the data
    train_eigenvalues = False
    test_classifier = False
    test_data_informed_classifier = False
    test_data_informed_classifier_2d = False
    test_loop_hafnian_wsm = True
    old_way = False
    fineness = 100

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
    # eigenvalue_generator = SmoothExponentialFasshauer(order)
    hyperparameters = GCPOSEHyperparameters(
        basis=ortho_basis, dimension=dimension
    )
    prior_mean = torch.tensor(0.0)
    alpha = torch.tensor(1.5)
    beta = torch.tensor(4.0)
    nu = torch.tensor(0.01)
    prior_parameters = PriorParameters(prior_mean, alpha, beta, nu)
    gcp_ose_1 = BayesianOrthogonalSeriesCoxProcess(
        hyperparameters, prior_parameters
    )
    gcp_ose_2 = BayesianOrthogonalSeriesCoxProcess(
        hyperparameters, prior_parameters
    )

    gcp_ose_1.add_data(class_1_data)
    gcp_ose_2.add_data(class_2_data)

    posterior_mean_1 = gcp_ose_1._get_posterior_mean()
    posterior_mean_2 = gcp_ose_2._get_posterior_mean()

    if plot_intensity:
        plt.plot(x.cpu().numpy(), posterior_mean_1(x).cpu().numpy())
        plt.plot(x.cpu().numpy(), intensity_1(x).cpu().numpy())
        plt.show()
        plt.plot(x.cpu().numpy(), posterior_mean_2(x).cpu().numpy())
        plt.plot(x.cpu().numpy(), intensity_2(x).cpu().numpy())
        plt.show()

    if test_classifier:
        test_axis = torch.linspace(0.0, max_time, fineness)
        # prior parameters
        prior_mean = torch.tensor(0.0)
        alpha = torch.tensor(1.5)
        beta = torch.tensor(4.0)
        nu = torch.tensor(0.01)
        prior_parameters_1 = PriorParameters(prior_mean, alpha, beta, nu)
        prior_parameters_2 = PriorParameters(prior_mean, alpha, beta, nu)

        hyperparameters_1 = GCPOSEHyperparameters(
            basis=ortho_basis, dimension=1
        )
        hyperparameters_2 = GCPOSEHyperparameters(
            basis=ortho_basis, dimension=1
        )

        classifier = GCPClassifier(
            2,
            [prior_parameters_1, prior_parameters_2],
            [hyperparameters_1, hyperparameters_2],
        )
        # add the data
        classifier.add_data(class_1_data, 0)
        classifier.add_data(class_2_data, 1)

        # now predict
        class_probs = classifier.predict_point(test_axis).cpu().numpy()
        print("ask class_probs for information about the Cox processes")
        breakpoint()
        plt.plot(test_axis.cpu().numpy(), class_probs[:, 0])
        plt.plot(test_axis.cpu().numpy(), class_probs[:, 1])
        plt.show()

    if test_data_informed_classifier:
        print("Beginning the data informed classifier phase!")
        test_axis = torch.linspace(0.0, max_time, fineness)

        # prior parameters
        prior_parameters_1 = DataInformedPriorParameters(nu)
        prior_parameters_2 = DataInformedPriorParameters(nu)

        hyperparameters_1 = GCPOSEHyperparameters(
            basis=ortho_basis, dimension=1
        )
        hyperparameters_2 = GCPOSEHyperparameters(
            basis=ortho_basis, dimension=1
        )

        classifier = GCPClassifier(
            2,
            [prior_parameters_1, prior_parameters_2],
            [hyperparameters_1, hyperparameters_2],
            data_informed=True,
        )

        # add the data
        classifier.add_data(class_1_data, 0)
        classifier.add_data(class_2_data, 1)

        # now predict
        class_probs = classifier.predict_point(test_axis).cpu().numpy()
        print("ask class_probs for information about the Cox processes")
        breakpoint()
        plt.plot(test_axis.cpu().numpy(), class_probs[:, 0])
        plt.plot(test_axis.cpu().numpy(), class_probs[:, 1])
        plt.show()

    if test_data_informed_classifier_2d:
        train_class_probs = True

        fineness = 300
        # get the appropriate data
        num_points = 150
        class_1_data, class_2_data = generate_points_in_square(num_points)

        order = 16
        dimension = 2
        basis_functions = [standard_chebyshev_basis] * 2
        max_time = 1.0
        extra_window = 0.05
        parameters: dict = [
            {
                "lower_bound": 0.0 - extra_window,
                "upper_bound": max_time + extra_window,
            },
        ] * 2
        ortho_basis = Basis(basis_functions, dimension, order, parameters)
        print("Beginning the data informed classifier phase!")
        test_axis = torch.linspace(0.0, max_time, fineness)
        test_axes_x, test_axes_y = torch.meshgrid(test_axis, test_axis)
        test_points = torch.vstack(
            (test_axes_x.ravel(), test_axes_y.ravel())
        ).t()

        # prior parameters
        prior_parameters_1 = DataInformedPriorParameters(nu)
        prior_parameters_2 = DataInformedPriorParameters(nu)

        hyperparameters_1 = GCPOSEHyperparameters(
            basis=ortho_basis, dimension=2
        )
        hyperparameters_2 = GCPOSEHyperparameters(
            basis=ortho_basis, dimension=2
        )

        classifier = GCPClassifier(
            2,
            [prior_parameters_1, prior_parameters_2],
            [hyperparameters_1, hyperparameters_2],
            data_informed=True,
        )

        # add the data
        classifier.add_data(class_1_data, 0)
        classifier.add_data(class_2_data, 1)

        # now predict
        if train_class_probs:
            class_probs = classifier.predict_point(test_points).cpu().numpy()
            # save the class probs so it is not so slow each time
            torch.save(class_probs, "class_probs.pt")
        else:
            class_probs = torch.load("class_probs.pt")

        print("ask class_probs for information about the Cox processes")
        # breakpoint()
        plt.contourf(
            test_axes_x.cpu().numpy(),
            test_axes_y.cpu().numpy(),
            class_probs[:, 0].reshape(test_axes_x.shape),
        )
        plt.show()

    if test_loop_hafnian_wsm:
        fineness = 300
        # get the appropriate data
        num_points = 150
        class_1_data, class_2_data = generate_points_in_square(num_points)

        order = 10
        dimension = 2
        basis_functions = [standard_chebyshev_basis] * 2
        max_time = 1.0
        extra_window = 0.05
        parameters: dict = [
            {
                "lower_bound": 0.0 - extra_window,
                "upper_bound": max_time + extra_window,
            },
        ] * 2
        ortho_basis = Basis(basis_functions, dimension, order, parameters)
        print("Beginning the data informed classifier phase!")
        test_axis = torch.linspace(0.0, max_time, fineness)
        test_axes_x, test_axes_y = torch.meshgrid(test_axis, test_axis)
        test_points = torch.vstack(
            (test_axes_x.ravel(), test_axes_y.ravel())
        ).t()

        # prior parameters
        prior_parameters_1 = DataInformedPriorParameters(nu)
        prior_parameters_2 = DataInformedPriorParameters(nu)

        hyperparameters_1 = GCPOSEHyperparameters(
            basis=ortho_basis, dimension=2
        )
        hyperparameters_2 = GCPOSEHyperparameters(
            basis=ortho_basis, dimension=2
        )

        classifier = GCPClassifier(
            2,
            [prior_parameters_1, prior_parameters_2],
            [hyperparameters_1, hyperparameters_2],
            data_informed=True,
        )

        # add the data
        classifier.add_data(class_1_data, 0)
        classifier.add_data(class_2_data, 1)
        relevant_cox_process = classifier.cox_processes[0]
        kernel_matrix = relevant_cox_process.get_kernel(
            relevant_cox_process.data_points, relevant_cox_process.data_points
        )
        eigenvalues = classifier.cox_processes[0].eigenvalues
        mean_function = classifier.cox_processes[0]._get_posterior_mean()

        # make it for class one
        # breakpoint()
        design_matrix = ortho_basis(relevant_cox_process.data_points)
        # print(design_matrix.shape)
        loop_hafnian_WSM = loop_hafnian_WSM(
            design_matrix,
            kernel_matrix,
            eigenvalues,
            mean_function(class_1_data),
            det_count=5000,
        )
