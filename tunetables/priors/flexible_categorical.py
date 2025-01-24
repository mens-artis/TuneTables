import random
import time

import torch
from torch import nn

from tunetables.utils import normalize_data, nan_handling_missing_for_unknown_reason_value, \
    nan_handling_missing_for_no_reason_value, nan_handling_missing_for_a_reason_value, to_ranking_low_mem, \
    remove_outliers, normalize_by_used_features_f
from .utils import get_batch_to_dataloader
from .utils import randomize_classes, CategoricalActivation
from .utils import uniform_int_sampler_f

time_it = False


class BalancedBinarize(nn.Module):
    """
    A module that binarizes input data based on its median value.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Binarize the input tensor based on its median value.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Binarized tensor (0.0 or 1.0)
        """
        return (x > torch.median(x)).float()


def class_sampler_f(min_, max_):
    """
    Creates a function that samples class numbers.

    Args:
        min_ (int): Minimum number of classes
        max_ (int): Maximum number of classes

    Returns:
        function: A function that returns either a random number between min_ and max_, or 2
    """

    def s():
        if random.random() > 0.5:
            return uniform_int_sampler_f(min_, max_)()
        return 2

    return s


class RegressionNormalized(nn.Module):
    """
    A module that normalizes regression data to the range [0, 1].
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Normalize the input tensor to the range [0, 1].

        Args:
            x (torch.Tensor): Input tensor of shape (T, B)

        Returns:
            torch.Tensor: Normalized tensor
        """
        # TODO: Normalize to -1, 1 or gaussian normal
        maxima = torch.max(x, 0)[0]
        minima = torch.min(x, 0)[0]
        norm = (x - minima) / (maxima - minima)
        return norm


class MulticlassRank(nn.Module):
    """
    A module that converts continuous data into multiclass rankings.
    """

    def __init__(self, num_classes, ordered_p=0.5):
        super().__init__()
        self.num_classes = class_sampler_f(2, num_classes)()
        self.ordered_p = ordered_p

    def forward(self, x):
        """
        Convert continuous data into multiclass rankings.

        Args:
            x (torch.Tensor): Input tensor of shape (T, B, H)

        Returns:
            torch.Tensor: Tensor of multiclass rankings
        """
        # CAUTION: This samples the same idx in sequence for each class boundary in a batch
        class_boundaries = torch.randint(0, x.shape[0], (self.num_classes - 1,))
        class_boundaries = x[class_boundaries].unsqueeze(1)

        d = (x > class_boundaries).sum(axis=0)

        # Randomize and reverse classes based on probabilities
        randomized_classes = torch.rand((d.shape[1],)) > self.ordered_p
        d[:, randomized_classes] = randomize_classes(d[:, randomized_classes], self.num_classes)
        reverse_classes = torch.rand((d.shape[1],)) > 0.5
        d[:, reverse_classes] = self.num_classes - 1 - d[:, reverse_classes]
        return d


class MulticlassValue(nn.Module):
    """
    A module that converts continuous data into multiclass values.
    """

    def __init__(self, num_classes, ordered_p=0.5):
        super().__init__()
        self.num_classes = class_sampler_f(2, num_classes)()
        self.classes = nn.Parameter(torch.randn(self.num_classes - 1), requires_grad=False)
        self.ordered_p = ordered_p

    def forward(self, x):
        """
        Convert continuous data into multiclass values.

        Args:
            x (torch.Tensor): Input tensor of shape (T, B, H)

        Returns:
            torch.Tensor: Tensor of multiclass values
        """
        d = (x > (self.classes.unsqueeze(-1).unsqueeze(-1))).sum(axis=0)

        # Randomize and reverse classes based on probabilities
        randomized_classes = torch.rand((d.shape[1],)) > self.ordered_p
        d[:, randomized_classes] = randomize_classes(d[:, randomized_classes], self.num_classes)
        reverse_classes = torch.rand((d.shape[1],)) > 0.5
        d[:, reverse_classes] = self.num_classes - 1 - d[:, reverse_classes]
        return d


class MulticlassMultiNode(nn.Module):
    """
    A module that converts continuous data into multiclass values using multiple nodes.
    """

    def __init__(self, num_classes, ordered_p=0.5):
        super().__init__()
        self.num_classes = class_sampler_f(2, num_classes)()
        self.classes = nn.Parameter(torch.randn(num_classes - 1), requires_grad=False)
        self.alt_multi_class = MulticlassValue(num_classes, ordered_p)

    def forward(self, x):
        """
        Convert continuous data into multiclass values using multiple nodes.

        Args:
            x (torch.Tensor): Input tensor of shape (T, B, H) or (T, B)

        Returns:
            torch.Tensor: Tensor of multiclass values
        """
        if len(x.shape) == 2:
            return self.alt_multi_class(x)
        T = 3
        x[torch.isnan(x)] = 0.00001
        d = torch.multinomial(
            torch.pow(0.00001 + torch.sigmoid(x[:, :, 0:self.num_classes]).reshape(-1, self.num_classes), T), 1,
            replacement=True).reshape(x.shape[0], x.shape[1])
        return d


class FlexibleCategorical(torch.nn.Module):
    """
    A flexible categorical model that can handle various data transformations and classifications.
    """

    def __init__(self, get_batch, hyperparameters, args):
        super(FlexibleCategorical, self).__init__()

        # Initialize hyperparameters and arguments
        self.h = {k: hyperparameters[k]() if callable(hyperparameters[k]) else hyperparameters[k] for k in
                  hyperparameters.keys()}
        self.args = args
        self.args_passed = {**self.args}
        self.args_passed.update({'num_features': self.h['num_features_used']})
        self.get_batch = get_batch

        # Set up class assigner based on hyperparameters
        if self.h['num_classes'] == 0:
            self.class_assigner = RegressionNormalized()
        else:
            if self.h['num_classes'] > 1 and not self.h['balanced']:
                if self.h['multiclass_type'] == 'rank':
                    self.class_assigner = MulticlassRank(self.h['num_classes'],
                                                         ordered_p=self.h['output_multiclass_ordered_p'])
                elif self.h['multiclass_type'] == 'value':
                    self.class_assigner = MulticlassValue(self.h['num_classes'],
                                                          ordered_p=self.h['output_multiclass_ordered_p'])
                elif self.h['multiclass_type'] == 'multi_node':
                    self.class_assigner = MulticlassMultiNode(self.h['num_classes'])
                else:
                    raise ValueError("Unknown Multiclass type")
            elif self.h['num_classes'] == 2 and self.h['balanced']:
                self.class_assigner = BalancedBinarize()
            elif self.h['num_classes'] > 2 and self.h['balanced']:
                raise NotImplementedError("Balanced multiclass training is not possible")

    def drop_for_reason(self, x, v):
        """
        Introduce missing values for a reason.

        Args:
            x (torch.Tensor): Input tensor
            v: Value to replace missing data

        Returns:
            torch.Tensor: Tensor with introduced missing values
        """
        nan_prob_sampler = CategoricalActivation(ordered_p=0.0, categorical_p=1.0, keep_activation_size=False,
                                                 num_classes_sampler=lambda: 20)
        d = nan_prob_sampler(x)
        # TODO: Make a different ordering for each activation
        x[d < torch.rand((1,), device=x.device) * 20 * self.h['nan_prob_no_reason'] * random.random()] = v
        return x

    def drop_for_no_reason(self, x, v):
        """
        Introduce missing values for no reason.

        Args:
            x (torch.Tensor): Input tensor
            v: Value to replace missing data

        Returns:
            torch.Tensor: Tensor with introduced missing values
        """
        x[torch.rand(x.shape, device=self.args['device']) < random.random() * self.h['nan_prob_no_reason']] = v
        return x

    def forward(self, batch_size):
        """
        Forward pass of the model.

        This method generates a batch of data, applies various transformations and normalizations,
        and prepares it for model processing.

        Args:
            batch_size (int): The size of the batch to generate.

        Returns:
            tuple: A tuple containing three tensors (x, y, y_), where:
                x (torch.Tensor): The input features.
                y (torch.Tensor): The target values.
                y_ (torch.Tensor): A copy of the target values.
        """
        start = time.time()
        # Generate a batch of data
        x, y, y_ = self.get_batch(hyperparameters=self.h, **self.args_passed)
        if time_it:
            print('Flex Forward Block 1', round(time.time() - start, 3))

        start = time.time()

        # Apply NaN transformations based on probabilities
        if self.h['nan_prob_no_reason'] + self.h['nan_prob_a_reason'] + self.h[
            'nan_prob_unknown_reason'] > 0 and random.random() > 0.5:
            if random.random() < self.h['nan_prob_no_reason']:
                # Missing for no reason
                x = self.drop_for_no_reason(x, nan_handling_missing_for_no_reason_value(self.h['set_value_to_nan']))

            if self.h['nan_prob_a_reason'] > 0 and random.random() > 0.5:
                # Missing for a reason
                x = self.drop_for_reason(x, nan_handling_missing_for_a_reason_value(self.h['set_value_to_nan']))

            if self.h['nan_prob_unknown_reason'] > 0:
                # Missing for unknown reason
                if random.random() < self.h['nan_prob_unknown_reason_reason_prior']:
                    x = self.drop_for_no_reason(x, nan_handling_missing_for_unknown_reason_value(
                        self.h['set_value_to_nan']))
                else:
                    x = self.drop_for_reason(x,
                                             nan_handling_missing_for_unknown_reason_value(self.h['set_value_to_nan']))

        # Apply categorical feature transformation
        if 'categorical_feature_p' in self.h and random.random() < self.h['categorical_feature_p']:
            p = random.random()
            for col in range(x.shape[2]):
                num_unique_features = max(round(random.gammavariate(1, 10)), 2)
                m = MulticlassRank(num_unique_features, ordered_p=0.3)
                if random.random() < p:
                    x[:, :, col] = m(x[:, :, col])

        if time_it:
            print('Flex Forward Block 2', round(time.time() - start, 3))
            start = time.time()

        # Normalize data
        if self.h['normalize_to_ranking']:
            x = to_ranking_low_mem(x)
        else:
            x = remove_outliers(x)
        x, y = normalize_data(x), normalize_data(y)

        if time_it:
            print('Flex Forward Block 3', round(time.time() - start, 3))
            start = time.time()

        # Cast to classification if enabled
        y = self.class_assigner(y).float()

        if time_it:
            print('Flex Forward Block 4', round(time.time() - start, 3))
            start = time.time()

        # Normalize by used features if enabled
        if self.h['normalize_by_used_features']:
            x = normalize_by_used_features_f(x, self.h['num_features_used'], self.args['num_features'],
                                             normalize_with_sqrt=self.h.get('normalize_with_sqrt', False))
        if time_it:
            print('Flex Forward Block 5', round(time.time() - start, 3))

        start = time.time()
        # Append empty features if enabled
        x = torch.cat([x, torch.zeros((x.shape[0], x.shape[1], self.args['num_features'] - self.h['num_features_used']),
                                      device=self.args['device'])], -1)
        if time_it:
            print('Flex Forward Block 6', round(time.time() - start, 3))

        # Check for NaNs in target
        if torch.isnan(y).sum() > 0:
            print('Nans in target!')

        # Ensure compatibility of train and eval targets
        if self.h['check_is_compatible']:
            for b in range(y.shape[1]):
                is_compatible, N = False, 0
                while not is_compatible and N < 10:
                    targets_in_train = torch.unique(y[:self.args['single_eval_pos'], b], sorted=True)
                    targets_in_eval = torch.unique(y[self.args['single_eval_pos']:, b], sorted=True)

                    is_compatible = len(targets_in_train) == len(targets_in_eval) and (
                            targets_in_train == targets_in_eval).all() and len(targets_in_train) > 1

                    if not is_compatible:
                        randperm = torch.randperm(x.shape[0])
                        x[:, b], y[:, b] = x[randperm, b], y[randperm, b]
                    N = N + 1
                if not is_compatible:
                    y[:, b] = -100  # Set to ignore_index for CE loss

        # Normalize labels if enabled
        if self.h['normalize_labels']:
            for b in range(y.shape[1]):
                valid_labels = y[:, b] != -100
                if self.h.get('normalize_ignore_label_too', False):
                    valid_labels[:] = True
                y[valid_labels, b] = (y[valid_labels, b] > y[valid_labels, b].unique().unsqueeze(1)).sum(
                    axis=0).unsqueeze(0).float()

                if y[valid_labels, b].numel() != 0 and self.h.get('rotate_normalized_labels', True):
                    num_classes_float = (y[valid_labels, b].max() + 1).cpu()
                    num_classes = num_classes_float.int().item()
                    assert num_classes == num_classes_float.item()
                    random_shift = torch.randint(0, num_classes, (1,), device=self.args['device'])
                    y[valid_labels, b] = (y[valid_labels, b] + random_shift) % num_classes

        return x, y, y  # x.shape = (T,B,H)


@torch.no_grad()
def get_batch(batch_size, seq_len, num_features, get_batch, device, hyperparameters=None, batch_size_per_gp_sample=None,
              **kwargs):
    """
    Generate a batch of data using multiple FlexibleCategorical models.

    Args:
        batch_size (int): Total batch size.
        seq_len (int): Sequence length.
        num_features (int): Number of features.
        get_batch (function): Function to get a single batch.
        device (torch.device): Device to use for computations.
        hyperparameters (dict, optional): Hyperparameters for the models.
        batch_size_per_gp_sample (int, optional): Batch size for each GP sample.
        **kwargs: Additional arguments to pass to the get_batch function.

    Returns:
        tuple: A tuple containing three tensors (x, y, y_), where:
            x (torch.Tensor): The input features.
            y (torch.Tensor): The target values.
            y_ (torch.Tensor): A copy of the target values.
    """
    batch_size_per_gp_sample = batch_size_per_gp_sample or (min(32, batch_size))
    num_models = batch_size // batch_size_per_gp_sample
    assert num_models > 0, f'Batch size ({batch_size}) is too small for batch_size_per_gp_sample ({batch_size_per_gp_sample})'
    assert num_models * batch_size_per_gp_sample == batch_size, f'Batch size ({batch_size}) not divisible by batch_size_per_gp_sample ({batch_size_per_gp_sample})'

    # Sample one seq_len for entire batch
    seq_len = hyperparameters['seq_len_used']() if callable(hyperparameters['seq_len_used']) else seq_len

    args = {'device': device, 'seq_len': seq_len, 'num_features': num_features, 'batch_size': batch_size_per_gp_sample,
            **kwargs}

    # Create multiple FlexibleCategorical models
    models = [FlexibleCategorical(get_batch, hyperparameters, args).to(device) for _ in range(num_models)]

    # Generate samples from each model
    sample = [model(batch_size=batch_size_per_gp_sample) for model in models]

    # Combine samples from all models
    x, y, y_ = zip(*sample)
    x, y, y_ = torch.cat(x, 1).detach(), torch.cat(y, 1).detach(), torch.cat(y_, 1).detach()

    return x, y, y_


# Convert get_batch function to a DataLoader
DataLoader = get_batch_to_dataloader(get_batch)
