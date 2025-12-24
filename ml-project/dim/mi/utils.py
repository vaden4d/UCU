'''

Various miscellaneous modules from cortex_DIM.

'''

import math

import torch
import torch.nn.functional as F


def log_sum_exp(x, axis=None):
    """Log sum exp function
    Args:
        x: Input.
        axis: Axis over which to perform sum.
    Returns:
        torch.Tensor: log sum exp
    """
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y


def raise_measure_error(measure):
    supported_measures = ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
    raise NotImplementedError(
        'Measure `{}` not supported. Supported: {}'.format(measure,
                                                           supported_measures))


def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.
    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        raise_measure_error(measure)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.
    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        raise_measure_error(measure)

    if average:
        return Eq.mean()
    else:
        return Eq

class View(torch.nn.Module):
    """Basic reshape module.
    """
    def __init__(self, *shape):
        """
        Args:
            *shape: Input shape.
        """
        super().__init__()
        self.shape = shape

    def forward(self, input):
        """Reshapes tensor.
        Args:
            input: Input tensor.
        Returns:
            torch.Tensor: Flattened tensor.
        """
        return input.view(*self.shape)


class Unfold(torch.nn.Module):
    """Module for unfolding tensor.
    Performs strided crops on 2d (image) tensors. Stride is assumed to be half the crop size.
    """
    def __init__(self, img_size, fold_size):
        """
        Args:
            img_size: Input size.
            fold_size: Crop size.
        """
        super().__init__()

        fold_stride = fold_size // 2
        self.fold_size = fold_size
        self.fold_stride = fold_stride
        self.n_locs = 2 * (img_size // fold_size) - 1
        self.unfold = torch.nn.Unfold((self.fold_size, self.fold_size),
                                      stride=(self.fold_stride, self.fold_stride))

    def forward(self, x):
        """Unfolds tensor.
        Args:
            x: Input tensor.
        Returns:
            torch.Tensor: Unfolded tensor.
        """
        N = x.size(0)
        x = self.unfold(x).reshape(N, -1, self.fold_size, self.fold_size, self.n_locs * self.n_locs)\
            .permute(0, 4, 1, 2, 3)\
            .reshape(N * self.n_locs * self.n_locs, -1, self.fold_size, self.fold_size)
        return x


class Fold(torch.nn.Module):
    """Module (re)folding tensor.
    Undoes the strided crops above. Works only on 1x1.
    """
    def __init__(self, img_size, fold_size):
        """
        Args:
            img_size: Images size.
            fold_size: Crop size.
        """
        super().__init__()
        self.n_locs = 2 * (img_size // fold_size) - 1

    def forward(self, x):
        """(Re)folds tensor.
        Args:
            x: Input tensor.
        Returns:
            torch.Tensor: Refolded tensor.
        """
        dim_c, dim_x, dim_y = x.size()[1:]
        x = x.reshape(-1, self.n_locs * self.n_locs, dim_c, dim_x * dim_y)
        x = x.reshape(-1, self.n_locs * self.n_locs, dim_c, dim_x * dim_y)\
            .permute(0, 2, 3, 1)\
            .reshape(-1, dim_c * dim_x * dim_y, self.n_locs, self.n_locs).contiguous()
        return x


class Permute(torch.nn.Module):
    """Module for permuting axes.
    """
    def __init__(self, *perm):
        """
        Args:
            *perm: Permute axes.
        """
        super().__init__()
        self.perm = perm

    def forward(self, input):
        """Permutes axes of tensor.
        Args:
            input: Input tensor.
        Returns:
            torch.Tensor: permuted tensor.
        """
        return input.permute(*self.perm)
