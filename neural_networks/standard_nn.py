import torch
from torch import nn
from tqdm import tqdm
from typing import Optional, Tuple
from collections import defaultdict
from .base_nets import BasePerceptron


class DeterministicPerceptron(BasePerceptron):
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
    ) -> torch.Tensor:
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
