import time
import torch
from torch import nn

# Attempt to import GPyTorch for Gaussian Processes; notify if not installed
try:
    import gpytorch
except ImportError:
    print("gpytorch not installed, please install it using 'pip install gpytorch' to enable fast_gp prior")

from .utils import get_batch_to_dataloader
from tunetables.utils import default_device


# Define a simple Gaussian Process model using exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    """
    A Gaussian Process model for regression using exact inference.

    Args:
        train_x (Tensor): Training input data.
        train_y (Tensor): Training output data.
        likelihood (GaussianLikelihood): The likelihood function for the model.
    """

    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()  # Mean function
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())  # Covariance function

    def forward(self, x):
        """
        Forward pass through the GP model.

        Args:
            x (Tensor): Input data for which to compute the output.

        Returns:
            MultivariateNormal: A distribution over the outputs.
        """
        mean_x = self.mean_module(x)  # Compute mean
        covar_x = self.covar_module(x)  # Compute covariance
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)  # Return multivariate normal distribution


def get_model(x, y, hyperparameters):
    """
    Create a Gaussian Process model and its likelihood.

    Args:
        x (Tensor): Input training data.
        y (Tensor): Output training data.
        hyperparameters (dict): Hyperparameters for the model.

    Returns:
        tuple: A tuple containing the model and its likelihood.
    """
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1.e-9))
    model = ExactGPModel(x, y, likelihood)

    # Set hyperparameters for noise and kernel parameters
    model.likelihood.noise = torch.ones_like(model.likelihood.noise) * hyperparameters["noise"]
    model.covar_module.outputscale = torch.ones_like(model.covar_module.outputscale) * hyperparameters["outputscale"]
    model.covar_module.base_kernel.lengthscale = torch.ones_like(model.covar_module.base_kernel.lengthscale) * \
                                                 hyperparameters["lengthscale"]

    return model, likelihood


@torch.no_grad()
def get_batch(batch_size, seq_len, num_features, device=default_device, hyperparameters=None,
              equidistant_x=False, fix_x=None, **kwargs):
    """
    Generate a batch of input data for training/testing.

    Args:
        batch_size (int): Number of samples in the batch.
        seq_len (int): Length of each sample sequence.
        num_features (int): Number of features in each sample.
        device (torch.device): Device to which tensors will be moved.
        hyperparameters (dict or None): Hyperparameters for generating data.
        equidistant_x (bool): If True, generate equidistant x values.
        fix_x (Tensor or None): Fixed x values if provided.

    Returns:
        tuple: A tuple containing the generated input data and sampled outputs.
    """

    # Set default hyperparameters if none are provided
    if isinstance(hyperparameters, (tuple, list)):
        hyperparameters = {"noise": hyperparameters[0],
                           "outputscale": hyperparameters[1],
                           "lengthscale": hyperparameters[2],
                           "is_binary_classification": hyperparameters[3],
                           "normalize_by_used_features": hyperparameters[5],
                           "order_y": hyperparameters[6],
                           "sampling": hyperparameters[7]}
    elif hyperparameters is None:
        hyperparameters = {"noise": .1, "outputscale": .1, "lengthscale": .1}

    if 'verbose' in hyperparameters and hyperparameters['verbose']:
        print({"noise": hyperparameters['noise'],
               "outputscale": hyperparameters['outputscale'],
               "lengthscale": hyperparameters['lengthscale'],
               'batch_size': batch_size,
               'sampling': hyperparameters['sampling']})

    assert not (equidistant_x and (fix_x is not None)), "Cannot use equidistant_x with fixed x."

    with gpytorch.settings.fast_computations(*hyperparameters.get('fast_computations', (True, True, True))):

        # Generate input data based on specified conditions
        if equidistant_x:
            assert num_features == 1  # Equidistant x requires single feature
            x = torch.linspace(0, 1., seq_len).unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)
        elif fix_x is not None:
            assert fix_x.shape == (seq_len, num_features)
            x = fix_x.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        else:
            if hyperparameters.get('sampling', 'uniform') == 'uniform':
                x = torch.rand(batch_size, seq_len, num_features, device=device)  # Uniform sampling
            else:
                x = torch.randn(batch_size, seq_len, num_features, device=device)  # Normal sampling

        model, likelihood = get_model(x, torch.Tensor(), hyperparameters)
        model.to(device)

        is_fitted = False
        while not is_fitted:
            try:
                with gpytorch.settings.prior_mode(True):
                    model, likelihood = get_model(x, torch.Tensor(), hyperparameters)
                    model.to(device)

                    d = model(x)  # Forward pass through the GP model
                    d = likelihood(d)  # Apply likelihood to the predictions
                    sample = d.sample().transpose(0, 1)  # Sample from the distribution
                    is_fitted = True
            except RuntimeError:
                print('GP Fitting unsuccessful, retrying.. ')
                print(x)
                print(hyperparameters)

    if bool(torch.any(torch.isnan(x)).detach().cpu().numpy()):
        print({"noise": hyperparameters['noise'],
               "outputscale": hyperparameters['outputscale'],
               "lengthscale": hyperparameters['lengthscale'],
               'batch_size': batch_size})

    return x.transpose(0, 1), sample, sample  # Return transposed input and sampled outputs


DataLoader = get_batch_to_dataloader(get_batch)
DataLoader.num_outputs = 1


def get_model_on_device(x, y, hyperparameters, device):
    """
    Move the GP model to a specified device.

    Args:
        x (Tensor): Input training data.
        y (Tensor): Output training data.
        hyperparameters (dict): Hyperparameters for the model.
        device (torch.device): Device to which the model will be moved.

    Returns:
        tuple: A tuple containing the model and its likelihood on the specified device.
    """

    model, likelihood = get_model(x, y, hyperparameters)
    model.to(device)

    return model, likelihood


@torch.no_grad()
def evaluate(x, y, y_non_noisy, use_mse=False,
             hyperparameters={},
             get_model_on_device=get_model_on_device,
             device=default_device,
             step_size=1,
             start_pos=0):
    """
    Evaluate the GP model on a dataset.

    Args:
        x (Tensor): Input data for evaluation.
        y (Tensor): Actual output data for evaluation.
        y_non_noisy (Tensor): Non-noisy output data for reference.
        use_mse (bool): If True use MSE loss; otherwise use log probability loss.
        hyperparameters (dict): Hyperparameters for evaluation.
        get_model_on_device (function): Function to get the model on a specified device.
        device (torch.device): Device for computation.
        step_size (int): Step size for evaluation iterations.
        start_pos (int): Starting position in the input data.

    Returns:
         tuple: A tuple containing all losses after each time step,
                average losses after each time step,
                and total evaluation time.
     """

    start_time = time.time()

    losses_after_t = [.0] if start_pos == 0 else []
    all_losses_after_t = []

    with gpytorch.settings.fast_computations(*hyperparameters.get('fast_computations', (True, True, True))), \
            gpytorch.settings.fast_pred_var(False):

        # Iterate through input sequences for evaluation
        for t in range(max(start_pos, 1), len(x), step_size):
            loss_sum = 0.0
            model, likelihood = get_model_on_device(x[:t].transpose(0, 1),
                                                    y[:t].transpose(0, 1),
                                                    hyperparameters,
                                                    device)

            model.eval()  # Set the model to evaluation mode

            f = model(x[t].unsqueeze(1))  # Predict next value using current inputs
            l = likelihood(f)  # Apply likelihood to predictions

            means = l.mean.squeeze()  # Get mean predictions
            varis = l.covariance_matrix.squeeze()  # Get variances

            assert len(means.shape) == len(varis.shape) == 1
            assert len(means) == len(varis) == x.shape[1]

            if use_mse:
                c = nn.MSELoss(reduction='none')  # Mean Squared Error loss function
                ls = c(means, y[t])  # Compute MSE loss
            else:
                ls = -l.log_prob(y[t].unsqueeze(1))  # Compute log probability loss

            losses_after_t.append(ls.mean())  # Store average loss after this time step
            all_losses_after_t.append(ls.flatten())  # Store all losses after this time step

        return torch.stack(all_losses_after_t).to('cpu'), \
            torch.tensor(losses_after_t).to('cpu'), \
            time.time() - start_time


if __name__ == '__main__':
    hps = (.1, .1, .1)

    # Example usage of evaluate function with generated batch data
    for redo_idx in range(1):
        print(evaluate(*get_batch(1000, 10,
                                  hyperparameters=hps,
                                  num_features=10),
                       use_mse=False,
                       hyperparameters=hps))
