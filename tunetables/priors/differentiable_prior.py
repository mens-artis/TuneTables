import torch
from torch import nn
import math

from .utils import get_batch_to_dataloader
from tunetables.utils import default_device

from .utils import trunc_norm_sampler_f, beta_sampler_f, gamma_sampler_f, uniform_sampler_f, zipf_sampler_f, \
    uniform_int_sampler_f


def unpack_dict_of_tuples(d):
    """
    Converts a dictionary of tuples into a list of dictionaries.

    Args:
        d (dict): A dictionary where each value is a tuple.

    Returns:
        list: A list of dictionaries where each dictionary contains values
              from the corresponding position in the tuples.
              Example: {'a': (1, 2), 'b': (3, 4)} -> [{'a': 1, 'b': 3}, {'a': 2, 'b': 4}]
    """
    return [dict(zip(d.keys(), v)) for v in list(zip(*list(d.values())))]


class DifferentiableHyperparameter(nn.Module):
    """
    A class that represents a differentiable hyperparameter that can be sampled
    from various distributions. This class allows for dynamic sampling of hyperparameters
    during model training.

    Attributes:
        distribution (str): The type of distribution to sample from.
        embedding_dim (int): The dimensionality of the embedding.
        device (torch.device): The device to run computations on (CPU or GPU).
        hparams (dict): A dictionary holding hyperparameters for meta distributions.
        sampler (callable): A function that samples hyperparameter values.
    """

    def __init__(self, distribution, embedding_dim, device, **args):
        """
        Initializes the DifferentiableHyperparameter instance.

        Args:
            distribution (str): The distribution type ('uniform', 'beta', etc.).
            embedding_dim (int): The dimension of the embedding layer.
            device (torch.device): The device for tensor operations.
            **args: Additional arguments specific to the distribution.
        """
        super(DifferentiableHyperparameter, self).__init__()

        self.distribution = distribution
        self.embedding_dim = embedding_dim
        self.device = device

        # Set additional attributes from args
        for key in args:
            setattr(self, key, args[key])

        def get_sampler():
            """
            Determines the appropriate sampling function based on the specified distribution.

            Returns:
                tuple: A tuple containing the sampler function and its parameters
                       (min, max, mean, std).
            """
            if self.distribution == "uniform":
                if not hasattr(self, 'sample'):
                    return uniform_sampler_f(
                        self.min, self.max), self.min, self.max, (
                        self.max + self.min) / 2, math.sqrt(
                        1 / 12 * (self.max - self.min) * (self.max - self.min))
                else:
                    return lambda: self.sample, self.min, self.max, None, None
            elif self.distribution == "uniform_int":
                return uniform_int_sampler_f(
                    self.min, self.max), self.min, self.max, (
                    self.max + self.min) / 2, math.sqrt(
                    1 / 12 * (self.max - self.min) * (self.max - self.min))

        # Handle meta distributions with specific sampling logic
        if self.distribution.startswith("meta"):
            self.hparams = {}

            def sample_meta(f):
                """
                Samples hyperparameters for meta distributions.

                Args:
                    f (callable): A function that takes hyperparameter values as input.

                Returns:
                    tuple: Indicators and passed values after sampling.
                """
                indicators, passed = unpack_dict_of_tuples({hp: self.hparams[hp]() for hp in self.hparams})
                meta_passed = f(**passed)
                return indicators, meta_passed

            args_passed = {'device': device, 'embedding_dim': embedding_dim}

            # TODO: commit: The issue of "Unresolved reference
            #  'make_beta'" arises from the way the make_beta function is being referenced in the lambda function for
            #  self.sampler. Similar to the previous case with make_gamma, the make_beta function is defined locally
            #  within the __init__ method but is being passed as an argument to the lambda function, which can lead
            #  to scope resolution issues. Solution To resolve this, you can directly reference make_beta within the
            #  lambda function without passing it as an argument.
            # Define specific meta distributions and their samplers
            if self.distribution == "meta_beta":
                # Check if b and k are already defined; if not, create them as hyperparameters
                if hasattr(self, 'b') and hasattr(self, 'k'):
                    self.hparams = {'b': lambda: (None, self.b), 'k': lambda: (None, self.k)}
                else:
                    self.hparams = {
                        "b": DifferentiableHyperparameter(distribution="uniform", min=self.min,
                                                          max=self.max, **args_passed),
                        "k": DifferentiableHyperparameter(distribution="uniform", min=self.min,
                                                          max=self.max, **args_passed)
                    }

                def make_beta(b_value, k_value):
                    """Creates a beta sampler."""
                    return lambda: self.scale * beta_sampler_f(b_value, k_value)()

                # Assign the sampler for meta beta distribution
                self.sampler = lambda: sample_meta(make_beta)

            elif self.distribution == "meta_gamma":
                # Truncated normal where std and mean are drawn randomly logarithmically scaled
                if hasattr(self, 'alpha') and hasattr(self, 'scale'):
                    self.hparams = {'alpha': lambda: (None, self.alpha), 'scale': lambda: (None, self.scale)}
                else:
                    self.hparams = {
                        "alpha": DifferentiableHyperparameter(distribution="uniform", min=0.0,
                                                              max=math.log(self.max_alpha), **args_passed),
                        "scale": DifferentiableHyperparameter(distribution="uniform", min=0.0,
                                                              max=self.max_scale, **args_passed)
                    }

                # Todo: this code gives an unresolved reference make_gamma:
                #  def make_gamma(alpha_, scale_):
                #      """Creates a gamma sampler."""
                #      return lambda alpha=alpha_, scale=scale_: (
                #          self.lower_bound + round(gamma_sampler_f(math.exp(alpha),
                #                                                   scale / math.exp(alpha))()) if
                #          self.round else
                #          self.lower_bound + gamma_sampler_f(math.exp(alpha), scale / math.exp(alpha))()
                #      )
                #  Assign the sampler for meta gamma distribution
                #  self.sampler = lambda make_gamma=make_gamma: sample_meta(make_gamma)
                #  #
                #  The issue arises because the function make_gamma is defined inside the __init__
                #  method of the DifferentiableHyperparameter class, but it is being referenced in the lambda
                #  function for self.sampler as if it were a global variable. Python treats make_gamma as a local
                #  variable in this context, and it cannot be resolved when passed as an argument in the lambda
                #  function. To fix this issue, you can avoid passing make_gamma as an argument to the lambda
                #  function. Instead, you can directly reference it within the lambda function since it's already
                #  defined in the same scope. Here's how you can fix it:
                def make_gamma(alpha_, scale_):
                    """Creates a gamma sampler."""
                    return lambda alpha=alpha_, scale=scale_: (
                        self.lower_bound + round(gamma_sampler_f(math.exp(alpha),
                                                                 scale / math.exp(alpha))()) if
                        self.round else
                        self.lower_bound + gamma_sampler_f(math.exp(alpha), scale / math.exp(alpha))()
                    )

                # Assign the sampler for meta gamma distribution
                self.sampler = lambda: sample_meta(make_gamma)

            elif self.distribution == "meta_trunc_norm_log_scaled":
                # Ensure standard deviation limits are set
                self.min_std = getattr(self, 'min_std', 0.01)
                self.max_std = getattr(self, 'max_std', 1.0)

                if not hasattr(self, 'log_mean'):
                    # Create log mean and log std hyperparameters using uniform distributions
                    self.hparams = {
                        "log_mean": DifferentiableHyperparameter(distribution="uniform", min=math.log(self.min_mean),
                                                                 max=math.log(self.max_mean), **args_passed),
                        "log_std": DifferentiableHyperparameter(distribution="uniform", min=math.log(self.min_std),
                                                                max=math.log(self.max_std), **args_passed)
                    }
                else:
                    # Use predefined values for log_mean and log_std
                    self.hparams = {
                        'log_mean': lambda: (None, self.log_mean),
                        'log_std': lambda: (None, self.log_std)
                    }

                def make_trunc_norm(log_mean, log_std):
                    """Creates a truncated normal sampler."""
                    return (
                        lambda: (
                                self.lower_bound + round(trunc_norm_sampler_f(math.exp(log_mean),
                                                                              math.exp(log_mean) * math.exp(log_std))())
                        ) if
                        self.round else
                        lambda: (
                                self.lower_bound + trunc_norm_sampler_f(math.exp(log_mean),
                                                                        math.exp(log_mean) * math.exp(log_std))()
                        )
                    )

                # Assign the sampler for truncated normal distribution with log scaling
                self.sampler = lambda _make_trunc_norm=make_trunc_norm: sample_meta(_make_trunc_norm)

            elif self.distribution == "meta_trunc_norm":
                # Ensure standard deviation limits are set
                self.min_std = getattr(self, 'min_std', 0.01)
                self.max_std = getattr(self, 'max_std', 1.0)

                # Create mean and std hyperparameters using uniform distributions
                self.hparams = {
                    "mean": DifferentiableHyperparameter(distribution="uniform", min=self.min_mean,
                                                         max=self.max_mean, **args_passed),
                    "std": DifferentiableHyperparameter(distribution="uniform", min=self.min_std,
                                                        max=self.max_std, **args_passed)
                }

                def make_trunc_norm(mean, std):
                    """Creates a truncated normal sampler."""
                    return (
                        lambda: (
                            self.lower_bound + round(trunc_norm_sampler_f(mean,
                                                                          std)()) if
                            self.round else
                            (
                                lambda: (
                                        self.lower_bound + trunc_norm_sampler_f(mean,
                                                                                std)()
                                )
                            )
                        )
                    )

                # Assign the sampler for truncated normal distribution
                self.sampler = lambda: sample_meta(make_trunc_norm)

            elif self.distribution == "meta_choice":
                if hasattr(self, 'choice_1_weight'):
                    # Use predefined weights for choices
                    self.hparams = {
                        f'choice_{i}_weight': lambda: (None,
                                                       getattr(self,
                                                               f'choice_{i}_weight'))
                        for i in range(1,
                                       len(self.choice_values))
                    }
                else:
                    # Create weights using uniform distributions for choices
                    self.hparams = {
                        f"choice_{i}_weight": DifferentiableHyperparameter(distribution="uniform", min=-3.0,
                                                                           max=5.0,
                                                                           **args_passed)
                        for i in range(1,
                                       len(self.choice_values))
                    }

                def make_choice(**choices):
                    """Creates a choice sampler based on weighted probabilities."""
                    choices_tensor = torch.tensor([1.0] + [choices[i] for i in choices], dtype=torch.float)
                    weights = torch.softmax(choices_tensor, 0)  # create a tensor of weights
                    sample = torch.multinomial(weights,
                                               1,
                                               replacement=True).numpy()[0]
                    return self.choice_values[sample]

                # Assign the sampler for choice-based distributions
                self.sampler = lambda _make_choice=make_choice: sample_meta(_make_choice)

            elif self.distribution == "meta_choice_mixed":
                if hasattr(self, 'choice_1_weight'):
                    # Use predefined weights for mixed choices
                    self.hparams = {
                        f'choice_{i}_weight': lambda: (None,
                                                       getattr(self,
                                                               f'choice_{i}_weight'))
                        for i in range(1,
                                       len(self.choice_values))
                    }
                else:
                    # Create weights using uniform distributions for mixed choices
                    self.hparams = {
                        f"choice_{i}_weight": DifferentiableHyperparameter(distribution="uniform", min=-5.0,
                                                                           max=6.0,
                                                                           **args_passed)
                        for i in range(1,
                                       len(self.choice_values))
                    }

                def make_choice(**choices):
                    """Creates a mixed choice sampler based on weighted probabilities."""
                    weights_tensor = torch.softmax(
                        torch.tensor([1.0] + [choices[i] for i in choices], dtype=torch.float), 0)

                    def sample():
                        s = torch.multinomial(weights_tensor,
                                              1,
                                              replacement=True).numpy()[0]
                        return self.choice_values[s]()

                    return lambda: sample

                # Assign the sampler for mixed choice-based distributions
                self.sampler = lambda _make_choice=make_choice: sample_meta(_make_choice)

        else:
            def return_two(x, min_val=None, max_val=None, mean=None, std=None):
                """
                Normalizes the sampled value and returns it along with an indicator.

                Args:
                     x (float): The sampled value.
                     min_val (float): Minimum value of the range.
                     max_val (float): Maximum value of the range.
                     mean (float): Mean value used for normalization.
                     std (float): Standard deviation used for normalization.

                 Returns:
                     tuple: A tuple containing normalized indicator and sampled value.
                 """
                if mean is not None:
                    ind = (x - mean) / std  # Normalize indicator to [-1 , 1]
                else:
                    ind = None
                return ind, x

        # Get the appropriate sampler function and its parameters based on distribution type
        self.sampler_f, \
            self.sampler_min, \
            self.sampler_max, \
            self.sampler_mean, \
            self.sampler_std = get_sampler()

        # Define the main sampling function using return_two logic
        # This will normalize sampled values before returning them.
        self.sampler = lambda: return_two(self.sampler_f(),
                                          min_val=self.sampler_min,
                                          max_val=self.sampler_max,
                                          mean=self.sampler_mean,
                                          std=self.sampler_std)


def forward(self):
    """
    Forward pass method to sample hyperparameters.

    Returns:
         tuple: A tuple containing sampled hyperparameter value and its normalized indicator.
    """
    s, s_passed = self.sampler()  # Call the sampling function defined in __init__
    return s, s_passed  # Return sampled value and its indicator


class DifferentiableHyperparameterList(nn.Module):
    """
    A class that manages a list of differentiable hyperparameters.

    This class initializes multiple instances of `DifferentiableHyperparameter`
    from a provided dictionary of hyperparameters and allows for sampling
    and retrieving information about these hyperparameters.

    Attributes:
        device (torch.device): The device to run computations on (CPU or GPU).
        hyperparameters (nn.ModuleDict): A dictionary of differentiable hyperparameters.
    """

    def __init__(self, hyperparameters, embedding_dim, device):
        """
        Initializes the DifferentiableHyperparameterList instance.

        Args:
            hyperparameters (dict): A dictionary containing hyperparameter configurations.
            embedding_dim (int): The dimensionality of the embedding for each hyperparameter.
            device (torch.device): The device for tensor operations.
        """
        super().__init__()

        self.device = device
        # Filter out any hyperparameters that are None or False
        hyperparameters = {k: v for (k, v) in hyperparameters.items() if v}

        # Create a ModuleDict containing DifferentiableHyperparameter instances
        self.hyperparameters = nn.ModuleDict({
            hp: DifferentiableHyperparameter(embedding_dim=embedding_dim,
                                             name=hp,
                                             device=device,
                                             **hyperparameters[hp])
            for hp in hyperparameters
        })

    def get_hyperparameter_info(self):
        """
        Retrieves information about the hyperparameters.

        Returns:
            tuple: A tuple containing:
                - sampled_hyperparameters_keys: List of keys for sampled hyperparameters.
                - sampled_hyperparameters_f: List of functions for remapping sampled values.
        """
        sampled_hyperparameters_f, sampled_hyperparameters_keys = [], []

        def append_hp(hp_key, hp_val):
            """
            Appends information about a hyperparameter to the lists.

            Args:
                hp_key (str): The key/name of the hyperparameter.
                hp_val (DifferentiableHyperparameter): The hyperparameter object.
            """
            sampled_hyperparameters_keys.append(hp_key)
            # Extract sampling parameters for normalization
            s_min, s_max, s_mean, s_std = hp_val.sampler_min, hp_val.sampler_max, hp_val.sampler_mean, hp_val.sampler_std

            # Append functions for normalizing and denormalizing values
            sampled_hyperparameters_f.append((lambda x: (x - s_mean) / s_std,
                                              lambda y: (y * s_std) + s_mean))

        # Iterate through all defined hyperparameters
        for hp in self.hyperparameters:
            hp_val = self.hyperparameters[hp]
            if hasattr(hp_val, 'hparams'):
                # If the hyperparameter has sub-hyperparameters, append them as well
                for hp_ in hp_val.hparams:
                    append_hp(f'{hp}_{hp_}', hp_val.hparams[hp_])
            else:
                append_hp(hp, hp_val)

        return sampled_hyperparameters_keys, sampled_hyperparameters_f

    def sample_parameter_object(self):
        """
        Samples values from all differentiable hyperparameters.

        Returns:
            tuple: A tuple containing:
                - s_passed: A dictionary of values passed to the model from the sampled parameters.
                - sampled_hyperparameters: A dictionary of sampled hyperparameter values.
        """
        sampled_hyperparameters, s_passed = {}, {}

        # Sample each hyperparameter and store results
        for hp in self.hyperparameters:
            sampled_hyperparameters_, s_passed_ = self.hyperparameters[hp]()
            s_passed[hp] = s_passed_

            if isinstance(sampled_hyperparameters_, dict):
                # Flatten nested dictionaries by concatenating keys
                sampled_hyperparameters_ = {hp + '_' + str(key): val for key, val in sampled_hyperparameters_.items()}
                sampled_hyperparameters.update(sampled_hyperparameters_)
            else:
                sampled_hyperparameters[hp] = sampled_hyperparameters_

        return s_passed, sampled_hyperparameters


class DifferentiablePrior(nn.Module):
    """
    A class representing a prior distribution over differentiable hyperparameters.

    This class uses a batch function to generate data while sampling from
    differentiable hyperparameter distributions.

    Attributes:
        h (dict): Base hyperparameter configuration.
        args (dict): Additional arguments for batch generation.
        get_batch (callable): Function to retrieve batches of data.
        differentiable_hyperparameters (DifferentiableHyperparameterList):
            Instance managing differentiable hyperparameters.
    """

    def __init__(self, get_batch, hyperparameters, differentiable_hyperparameters, args):
        """
        Initializes the DifferentiablePrior instance.

        Args:
            get_batch (callable): Function to retrieve batches of data.
            hyperparameters (dict): Base configuration for model parameters.
            differentiable_hyperparameters (dict): Configuration for differentiable hyperparameters.
            args (dict): Additional arguments for batch generation.
        """
        super(DifferentiablePrior, self).__init__()

        self.h = hyperparameters
        self.args = args
        self.get_batch = get_batch

        # Initialize the list of differentiable hyperparameters
        self.differentiable_hyperparameters = DifferentiableHyperparameterList(
            differentiable_hyperparameters,
            embedding_dim=self.h['emsize'],
            device=self.args['device']
        )

    def forward(self):
        """
        Forward pass method to sample from the prior distribution.

        Returns:
             tuple: A tuple containing:
                 - x: Input data generated from the batch function.
                 - y: Target output from the batch function.
                 - y_: Additional output from the batch function.
                 - sampled_hyperparameters_indicators: Indicators for the sampled hyperparameter values.
        """

        # Sample hyperparameters and their indicators
        sampled_hyperparameters_passed, sampled_hyperparameters_indicators = self.differentiable_hyperparameters.sample_parameter_object()

        # Combine base and sampled hyperparameter configurations
        hyperparameters = {**self.h, **sampled_hyperparameters_passed}

        # Retrieve a batch of data using combined configurations
        x, y, y_ = self.get_batch(hyperparameters=hyperparameters, **self.args)

        return x, y, y_, sampled_hyperparameters_indicators


# TODO: Make this a class that keeps objects
@torch.no_grad()
def get_batch(batch_size, seq_len, num_features, get_batch,
              device=default_device,
              differentiable_hyperparameters={},
              hyperparameters=None,
              batch_size_per_gp_sample=None,
              **kwargs):
    """
    Generates a batch of data by sampling from multiple models with differentiable prior distributions.

    Args:
         batch_size (int): Total number of samples to generate.
         seq_len (int): Length of each sequence in the batch.
         num_features (int): Number of features in each sample.
         get_batch (callable): Function to retrieve batches of data.
         device (torch.device): The device to run computations on (CPU or GPU).
         differentiable_hyperparameters (dict): Configuration for differentiable hyperparameters.
         hyperparameters (dict): Base configuration for model parameters.
         batch_size_per_gp_sample (int): Number of samples per model instance. Defaults to min(64, batch_size).
         **kwargs: Additional arguments passed to `get_batch`.

     Returns:
         tuple: A tuple containing:
             - x: Input data generated from the batch function.
             - y: Target output from the batch function.
             - y_: Additional output from the batch function.
             - packed_hyperparameters: Packed tensor of sampled differentiable hyperparameter values if applicable.
    """

    # Determine number of models based on specified batch size per sample
    batch_size_per_gp_sample = batch_size_per_gp_sample or min(64, batch_size)
    num_models = batch_size // batch_size_per_gp_sample

    # Ensure that total batch size is divisible by samples per model instance
    assert num_models * batch_size_per_gp_sample == batch_size, (
        f'Batch size ({batch_size}) not divisible by '
        f'batch_size_per_gp_sample ({batch_size_per_gp_sample})'
    )

    # Prepare arguments for model instantiation
    args = {'device': device,
            'seq_len': seq_len,
            'num_features': num_features,
            'batch_size': batch_size_per_gp_sample}

    args = {**kwargs, **args}

    # Create multiple instances of DifferentiablePrior models
    models = [DifferentiablePrior(get_batch,
                                  hyperparameters,
                                  differentiable_hyperparameters,
                                  args)
              for _ in range(num_models)]

    # Sample outputs from all models and aggregate results
    sample = sum([[model()] for model in models], [])

    x, y, y_, hyperparameter_dict = zip(*sample)

    if 'verbose' in hyperparameters and hyperparameters['verbose']:
        print('Hparams', hyperparameter_dict[0].keys())

    # Prepare a matrix to hold all sampled hyperparameter values
    hyperparameter_matrix = []

    # Extract and compile all relevant samples into a matrix format
    for batch in hyperparameter_dict:
        hyperparameter_matrix.append([batch[hp] for hp in batch])

    transposed_hyperparameter_matrix = list(zip(*hyperparameter_matrix))

    # Ensure consistency across samples regarding None values
    assert all([all([hp is None for hp in hp_]) or
                all([hp is not None for hp in hp_])
                for hp_ in transposed_hyperparameter_matrix]), (
        'It should always be the case that when a '
        'hyper-parameter is None once it is always None'
    )

    # Remove columns that are only None (i.e., not sampled)
    hyperparameter_matrix = [[hp for hp in hp_ if hp is not None]
                             for hp_ in hyperparameter_matrix]

    if len(hyperparameter_matrix[0]) > 0:
        # Create a tensor from the matrix of packed parameters
        packed_hyperparams = torch.tensor(hyperparameter_matrix)
        packed_hyperparams = torch.repeat_interleave(packed_hyperparams,
                                                     repeats=batch_size_per_gp_sample,
                                                     dim=0).detach()
    else:
        packed_hyperparams = None

    # Concatenate outputs into final tensors before returning them
    x, y, y_, packed_hyperparams = (
        torch.cat(x, 1).detach(),
        torch.cat(y, 1).detach(),
        torch.cat(y_, 1).detach(),
        packed_hyperparams)

    return x, y, y_, (
        packed_hyperparams if
        hyperparameters.get('differentiable_hps_as_style', True) else None)


DataLoader = get_batch_to_dataloader(get_batch)


def draw_random_style(dl, device):
    """
    Draws a random style embedding from a DataLoader.

    Args:
         dl (DataLoader): DataLoader instance providing batches of data.
         device (torch.device): The device to run computations on.

     Returns:
         torch.Tensor: A tensor containing a random style embedding from the DataLoader.
     """

    # Get one random sample from DataLoader and move it to specified device
    (hp_embedding, _, _), _, _ = next(iter(dl))
    return hp_embedding.to(device)[0:1, :]


def merge_style_with_info(diff_hparams_keys, diff_hparams_f,
                          style, transform=True):
    """
    Merges style information with differentiable hyperparameter values.

    Args:
         diff_hparams_keys (list): List containing keys/names of differentiable parameters.
         diff_hparams_f (list): List containing functions to transform parameter values.
         style (torch.Tensor): Style tensor to be merged with parameter values.
         transform (bool): Whether to apply transformation functions. Defaults to True.

     Returns:
         dict: A dictionary mapping parameter names to their corresponding transformed or raw values.
    """

    params = dict(zip(diff_hparams_keys,
                      zip(diff_hparams_f,
                          style.detach().cpu().numpy().tolist()[0])))

    def t(v):
        """Applies transformation if required."""
        if transform:
            return v[0][1](v[1])  # Apply denormalization function on style value
        else:
            return v[1]  # Return raw style value

    return {k: t(v) for k, v in params.items()}


def replace_differentiable_distributions(config):
    """
    Replaces entries in configuration with appropriate ConfigSpace HyperParameter objects.

    Args:
         config (dict): Configuration dictionary containing definitions
                        for differentiable distributions.

    Raises:
         ValueError: If an unknown distribution type is encountered during replacement process.
    """

    import ConfigSpace.hyperparameters as CSH

    diff_config = config['differentiable_hyper_parameters']

    # Iterate through each defined distribution and replace it with appropriate ConfigSpace types
    for name, diff_hp_dict in diff_config.items():
        distribution = diff_hp_dict['distribution']

        if distribution == 'uniform':
            diff_hp_dict['sample'] = CSH.UniformFloatHyperparameter(name, diff_hp_dict['min'], diff_hp_dict['max'])

        elif distribution == 'meta_beta':
            diff_hp_dict['k'] = CSH.UniformFloatHyperparameter(name + '_k', diff_hp_dict['min'], diff_hp_dict['max'])
            diff_hp_dict['b'] = CSH.UniformFloatHyperparameter(name + '_b', diff_hp_dict['min'], diff_hp_dict['max'])

        elif distribution == 'meta_gamma':
            diff_hp_dict['alpha'] = CSH.UniformFloatHyperparameter(name + '_k', 0.0,
                                                                   math.log(diff_hp_dict['max_alpha']))
            diff_hp_dict['scale'] = CSH.UniformFloatHyperparameter(name + '_b', 0.0, diff_hp_dict['max_scale'])

        elif distribution == 'meta_choice':
            for i in range(1, len(diff_hp_dict['choice_values'])):
                diff_hp_dict[f'choice_{i}_weight'] = CSH.UniformFloatHyperparameter(name + f'choice_{i}_weight', -3.0,
                                                                                    5.0)

        elif distribution == 'meta_choice_mixed':
            for i in range(1, len(diff_hp_dict['choice_values'])):
                diff_hp_dict[f'choice_{i}_weight'] = CSH.UniformFloatHyperparameter(name + f'choice_{i}_weight', -3.0,
                                                                                    5.0)

        elif distribution == 'meta_trunc_norm_log_scaled':
            diff_hp_dict['log_mean'] = CSH.UniformFloatHyperparameter(name + '_log_mean',
                                                                      math.log(diff_hp_dict['min_mean']),
                                                                      math.log(diff_hp_dict['max_mean']))

            min_std = diff_hp_dict.get('min_std', 0.1)
            max_std = diff_hp_dict.get('max_std', 1.0)

            diff_hp_dict['log_std'] = CSH.UniformFloatHyperparameter(name + '_log_std', math.log(min_std),
                                                                     math.log(max_std))
        else:
            raise ValueError(f'Unknown distribution {distribution}')
