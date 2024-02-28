import torch
from torch import nn
from tqdm import tqdm
from typing import Optional, Tuple
from collections import defaultdict

class LinearLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.bias = bias
        self.w = nn.Parameter(torch.randn((output_dim, input_dim + int(bias))) * 0.01)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 2
        assert x.shape[1] == self.input_dim

        if self.bias:
            ones = torch.ones((x.shape[0], 1))
            x = torch.cat((x, ones), dim=1)

        return x @ self.w.T


class BasePerceptron(nn.Module):
    def __init__(
        self,
        hidden_units: int,
        nonlinearity: str = "relu",
        prior_scale: float = 1.0,
        prior: str = "Gaussian",
        observation_noise: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        assert prior.lower() in ["gaussian", "laplacian", "laplace"]

        self.hidden_units = hidden_units
        self.prior_scale = prior_scale
        self.observation_noise = observation_noise
        self.bias = bias

        self.nonlinearity = None
        if nonlinearity.lower() == "relu":
            self.nonlinearity = nn.ReLU()
        elif nonlinearity.lower() == "elu":
            self.nonlinearity = nn.ELU()
        elif nonlinearity.lower() == "sigmoid":
            self.nonlinearity = nn.Sigmoid()
        elif nonlinearity.lower() == "leakyrelu":
            self.nonlinearity = nn.LeakyReLU(0.1)
        elif nonlinearity.lower() == "tanh":
            self.nonlinearity = nn.Tanh()
        else:
            raise NotImplementedError("Nonlinearity chosen not implemented")

        self.layer_1 = LinearLayer(1, hidden_units, bias=bias)
        self.layer_2 = LinearLayer(hidden_units, 1, bias=bias)

        if prior.lower() == "gaussian":
            self.layer_1_prior = torch.distributions.Normal(
                torch.zeros_like(self.layer_1.w),
                torch.ones_like(self.layer_1.w) * prior_scale,
            )
            self.layer_2_prior = torch.distributions.Normal(
                torch.zeros_like(self.layer_2.w),
                torch.ones_like(self.layer_2.w) * prior_scale,
            )
        elif prior.lower() in ["laplace", "laplacian"]:
            self.layer_1_prior = torch.distributions.Laplace(
                torch.zeros_like(self.layer_1.w),
                torch.ones_like(self.layer_1.w) * prior_scale,
            )
            self.layer_2_prior = torch.distributions.Laplace(
                torch.zeros_like(self.layer_2.w),
                torch.ones_like(self.layer_2.w) * prior_scale,
            )

        net = [self.layer_1, self.nonlinearity, self.layer_2]
        self.network = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        return self.network(x)
    
    
class MCMCPerceptron(BasePerceptron):
    def __init__(
        self,
        hidden_units: int,
        nonlinearity: str = "relu",
        prior_scale: float = 1.0,
        prior: str = "Gaussian",
        observation_noise: float = 0.1,
        bias: bool = True,
    ):
        super().__init__(
            hidden_units,
            nonlinearity=nonlinearity,
            prior_scale=prior_scale,
            prior=prior,
            observation_noise=observation_noise,
            bias=bias,
        )

    def params_to_sample(self) -> torch.Tensor:
        return torch.cat((self.layer_1.w.flatten(), self.layer_2.w.flatten()))

    def sample_to_params(self, sample: torch.Tensor):
        self.layer_1.w.data = sample[: self.layer_1.w.numel()].view(
            (self.layer_1.output_dim, self.layer_1.input_dim + int(self.layer_1.bias))
        )
        self.layer_2.w.data = sample[self.layer_1.w.numel() :].view(
            (self.layer_2.output_dim, self.layer_2.input_dim + int(self.layer_2.bias))
        )

    def get_potential(
        self, x: torch.Tensor, y: torch.Tensor, sample: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # U(q) = - log posterior, i.e. U(q) = - log(likelihood) - log(prior) + c

        # swap current parameters for parameters we are interested in evaluating
        if sample is not None:  #
            current_params = self.params_to_sample()
            self.sample_to_params(sample)

        if len(y.shape) == 1:
            y = y.unsqueeze(-1)

        preds = self(x)
        gaussian_likelihood = torch.distributions.Normal(preds, self.observation_noise)
        log_likelihood = gaussian_likelihood.log_prob(y).sum()
        log_prior = (
            self.layer_1_prior.log_prob(self.layer_1.w).sum()
            + self.layer_2_prior.log_prob(self.layer_2.w).sum()
        )

        U = -(log_likelihood + log_prior)

        # return model parameters to what they were before this function call
        if sample is not None:
            self.sample_to_params(current_params)

        return U

    def get_potential_grad(
        self, x: torch.Tensor, y: torch.Tensor, sample: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # compute gradients of potential w.r.t. parameters of interest (sample)
        U = self.get_potential(x, y, sample)
        self.zero_grad()
        U.backward()
        grads = torch.cat(
            (self.layer_1.w.grad.flatten(), self.layer_2.w.grad.flatten())
        )
        return grads