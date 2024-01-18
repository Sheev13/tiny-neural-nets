import torch
from torch import nn
from tqdm import tqdm
from typing import Optional

# TODO: refactor this into a BasePerceptron class and then DeterministicPerceptron, LangevinPerceptron, HamiltonianPerceptron on top


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

    def params_to_sample(self) -> torch.Tensor:
        return torch.cat((self.layer_1.w.flatten(), self.layer_2.w.flatten()))

    def sample_to_params(self, sample: torch.Tensor):
        self.layer_1.w.data = sample[: self.layer_1.w.numel()].view(
            (self.layer_1.output_dim, self.layer_1.input_dim + int(self.layer_1.bias))
        )
        self.layer_2.w.data = sample[self.layer_1.w.numel() :].view(
            (self.layer_2.output_dim, self.layer_2.input_dim + int(self.layer_2.bias))
        )

    def get_log_potential(
        self, x: torch.Tensor, y: torch.Tensor, sample: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # U(q) = - log posterior, i.e. U(q) = - log(likelihood) - log(prior) + c
        # this equation is conveniently already implemented in MAP_loss (up to negative sign and plus c term)

        # swap current parameters for parameters we are interested in evaluating
        if sample is not None:  #
            current_params = self.params_to_sample()
            self.sample_to_params(sample)

        U = -self.MAP_loss(x, y)

        # return model parameters to what they were before this function call
        if sample is not None:
            self.sample_to_params(current_params)

        return U

    def get_log_potential_grad(
        self, x: torch.Tensor, y: torch.Tensor, sample: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # compute gradients of potential w.r.t. parameters of interest (sample)
        U = self.get_log_potential(x, y, sample)
        self.zero_grad()
        U.backward()
        grads = torch.cat(
            (self.layer_1.w.grad.flatten(), self.layer_2.w.grad.flatten())
        )
        return grads

    def get_langevin_proposal(
        self, x: torch.Tensor, y: torch.Tensor, step_size: float = 1e-4
    ):
        # q* = q - step_size/2 * grad(U(q)) + sqrt(step_size) * randn
        params = self.params_to_sample()
        proposed_params = (
            params
            - (step_size / 2) * self.get_log_potential_grad(x, y)
            + torch.sqrt(torch.tensor(step_size)) * torch.randn_like(params)
        )
        return proposed_params

    def compute_log_langevin_proposal_prob(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sample: torch.Tensor,
        proposed_sample: torch.Tensor,
        step_size: float = 1e-4,
    ):
        # - 1/(2*stepsize)||q* - q - stepsize/2 * grad U(q)||^2
        grad_u = self.get_log_potential_grad(x, y, sample)
        norm = torch.linalg.vector_norm(
            proposed_sample - sample - (step_size / 2) * grad_u
        )
        log_prob = -(1 / (2 * step_size)) * torch.pow(norm, 2)
        return log_prob

    def compute_log_acceptance_prob(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sample: torch.Tensor,
        proposed_sample: torch.Tensor,
        step_size: float = 1e-4,
    ):
        u_q = self.get_log_potential(x, y, sample)
        u_q_star = self.get_log_potential(x, y, proposed_sample)
        q_to_q_star = self.compute_log_langevin_proposal_prob(
            x, y, sample, proposed_sample, step_size=step_size
        )
        q_star_to_q = self.compute_log_langevin_proposal_prob(
            x, y, proposed_sample, sample, step_size=step_size
        )
        return u_q_star - u_q + q_star_to_q - q_to_q_star

    def langevin_monte_carlo(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        simulation_steps: int = 1000,
        step_size: float = 1e-4,
        pbar: bool = True,
    ):
        sample_dimension = self.layer_1.w.numel() + self.layer_2.w.numel()
        posterior_samples = torch.zeros((simulation_steps, sample_dimension))
        acceptance_counter = 0

        for step in tqdm(range(simulation_steps), disable=not pbar):
            current_sample = self.params_to_sample()
            proposed_sample = self.get_langevin_proposal(x, y, step_size=step_size)
            log_alpha = self.compute_log_acceptance_prob(
                x, y, current_sample, proposed_sample, step_size=step_size
            )
            u = torch.rand((1,))
            if u < log_alpha.exp():
                # accept the sample
                posterior_samples[step] = proposed_sample
                self.sample_to_params(proposed_sample)
                acceptance_counter += 1
            else:
                # reject the sample
                posterior_samples[step] = current_sample

        average_acceptance = acceptance_counter / simulation_steps

        return posterior_samples, average_acceptance
