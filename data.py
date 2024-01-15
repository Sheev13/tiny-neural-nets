import torch
from typing import Optional, Tuple
from scipy.spatial import distance_matrix


def _sample_uniform_inputs(
    lower: float = -2.0, upper: float = 2.0, num: Optional[int] = None
) -> torch.Tensor:
    if not num:
        num = int(torch.randint(0, 30, (1,)))
    x = (torch.rand((num,)) * (upper - lower)) + lower
    return x


def _compute_covariance_matrix(
    x: torch.Tensor, lengthscale: float = 1.0
) -> torch.Tensor:
    if len(x.shape) < 2:
        x = x.unsqueeze(-1)
    dist_matrix = torch.tensor(distance_matrix(x, x, p=2))
    return torch.exp(-pow(dist_matrix, 2) / (2 * pow(lengthscale, 2)))


def _se_gp_prior_sample(x: torch.Tensor, lengthscale: float = 1.0) -> torch.Tensor:
    """Takes a sample function from a GP prior and returns the function evaluated at the inputs x"""
    num_points = x.shape[0]
    cov = _compute_covariance_matrix(x, lengthscale=lengthscale)
    randn_vector = torch.randn((num_points,))
    cov_cholesky = torch.linalg.cholesky(cov + torch.eye(num_points) * 1e-6)
    f_x = cov_cholesky @ randn_vector
    return f_x


def _get_noisy_observations(y: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
    num_points = y.shape[0]
    noise = torch.randn((num_points,)) * noise_level
    return y + noise


def generate_segp_dataset(
    input_lower: float = -2.0,
    input_upper: float = 2.0,
    num_points: int = 10,
    noise_level: float = 0.1,
    lengthscale: float = 1.0,
) -> Tuple[torch.Tensor]:
    """Pulls a randomly generated set of inputs and noisy squared-exponential
    GP-prior evaluated outputs out of thin air."""
    x = _sample_uniform_inputs(lower=input_lower, upper=input_upper, num=num_points)
    f_x = _se_gp_prior_sample(x, lengthscale=lengthscale)
    y = _get_noisy_observations(f_x, noise_level=noise_level)
    return x, y
