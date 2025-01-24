import torch
from .utils import get_batch_to_dataloader
from tunetables.utils import default_device


def get_batch(batch_size, seq_len, num_features, device=default_device,
              hyperparameters=None, batch_size_per_gp_sample=None, **kwargs):
    """
    Generate a batch of data for training or evaluation.

    Args:
        batch_size (int): Total number of samples in the batch.
        seq_len (int): Length of each sequence in the batch.
        num_features (int): Number of features for each data point.
        device (torch.device, optional): Device to store the tensors. Defaults to default_device.
        hyperparameters (dict, optional): Dictionary of hyperparameters for the model.
        batch_size_per_gp_sample (int, optional): Number of samples per GP (Gaussian Process) model.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple: A tuple containing three tensors (x, y, y_) representing input, target, and predicted values.

    Raises:
        AssertionError: If batch_size is not divisible by batch_size_per_gp_sample.
    """
    # Set default batch size per GP sample if not provided
    batch_size_per_gp_sample = batch_size_per_gp_sample or (min(64, batch_size))
    num_models = batch_size // batch_size_per_gp_sample

    # Ensure batch_size is divisible by batch_size_per_gp_sample
    assert num_models * batch_size_per_gp_sample == batch_size, (
        f'Batch size ({batch_size}) not divisible by batch_size_per_gp_sample ({batch_size_per_gp_sample})'
    )

    # Prepare arguments for get_batch functions
    args = {'device': device, 'seq_len': seq_len, 'num_features': num_features, 'batch_size': batch_size_per_gp_sample}

    # Extract prior bag get_batch functions and their weights from hyperparameters
    prior_bag_priors_get_batch = hyperparameters['prior_bag_get_batch']
    prior_bag_priors_p = [1.0] + [hyperparameters[f'prior_bag_exp_weights_{i}'] for i in
                                  range(1, len(prior_bag_priors_get_batch))]

    # Create a tensor of weights and sample batch assignments
    weights = torch.tensor(prior_bag_priors_p, dtype=torch.float)
    batch_assignments = torch.multinomial(torch.softmax(weights, 0), num_models, replacement=True).numpy()

    # Print debug information if verbose mode is enabled
    if 'verbose' in hyperparameters and hyperparameters['verbose']:
        print('PRIOR_BAG:', weights, batch_assignments)

    # Generate samples using the assigned get_batch functions
    sample = [prior_bag_priors_get_batch[int(prior_idx)](hyperparameters=hyperparameters, **args, **kwargs)
              for prior_idx in batch_assignments]

    # Unpack and concatenate the samples
    x, y, y_ = zip(*sample)
    x, y, y_ = (torch.cat(x, 1).detach(),
                torch.cat(y, 1).detach(),
                torch.cat(y_, 1).detach())

    return x, y, y_


# Create a DataLoader using the get_batch function
DataLoader = get_batch_to_dataloader(get_batch)
