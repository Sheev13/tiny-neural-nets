import torch
from torch import nn
from tqdm import tqdm

# TODO: add support for optional bias


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


class Perceptron(nn.Module):
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

    def ML_loss(  # maximum likelihood loss
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        if len(y.shape) == 1:
            y = y.unsqueeze(-1)

        preds = self(x)
        gaussian_likelihood = torch.distributions.Normal(preds, self.observation_noise)
        log_likelihood = gaussian_likelihood.log_prob(y).sum()

        return log_likelihood

    def MAP_loss(  # maximum a posteriori loss
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        if len(y.shape) == 1:
            y = y.unsqueeze(-1)

        preds = self(x)
        gaussian_likelihood = torch.distributions.Normal(preds, self.observation_noise)
        log_likelihood = gaussian_likelihood.log_prob(y).sum()
        log_prior = (
            self.layer_1_prior.log_prob(self.layer_1.w).sum()
            + self.layer_2_prior.log_prob(self.layer_2.w).sum()
        )

        return log_likelihood + log_prior

    def train(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_function: str = "ML",
        epochs: int = 100,
        algorithm: str = "Adam",
        learning_rate: float = 1e-2,
    ):
        assert loss_function.lower() in ["map", "ml"]
        assert algorithm.lower() in ["sgd", "adam"]

        if algorithm.lower() == "sgd":
            optimiser = torch.optim.SGD(self.parameters(), lr=learning_rate)
        elif algorithm.lower() == "adam":
            optimiser = torch.optim.Adam(self.parameters(), lr=learning_rate)

        loss_evolution = torch.zeros((epochs,))

        for epoch in tqdm(range(epochs)):
            optimiser.zero_grad()

            if loss_function.lower() == "ml":
                loss = -self.ML_loss(x, y)
            elif loss_function.lower() == "map":
                loss = -self.MAP_loss(x, y)

            loss_evolution[epoch] = -loss.item()

            loss.backward()
            optimiser.step()

        return loss_evolution
