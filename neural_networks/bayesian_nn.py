import torch
from torch import nn
from tqdm import tqdm
from typing import Optional, Tuple
from collections import defaultdict
from .base_nets import MCMCPerceptron


class LangevinPerceptron(MCMCPerceptron):
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

    def get_langevin_proposal(
        self, x: torch.Tensor, y: torch.Tensor, step_size: float = 1e-4
    ):
        # q* = q - step_size/2 * grad(U(q)) + sqrt(step_size) * randn
        params = self.params_to_sample()
        proposed_params = (
            params
            - (step_size / 2) * self.get_potential_grad(x, y)
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
        grad_u = self.get_potential_grad(x, y, sample)
        norm = torch.linalg.vector_norm(
            proposed_sample - sample - (step_size / 2) * grad_u
        )
        log_prob = -(1 / (2 * step_size)) * torch.pow(norm, 2)
        return log_prob

    def compute_log_lmc_acceptance(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sample: torch.Tensor,
        proposed_sample: torch.Tensor,
        step_size: float = 1e-4,
    ):
        u_q = self.get_potential(x, y, sample)
        u_q_star = self.get_potential(x, y, proposed_sample)
        q_to_q_star = self.compute_log_langevin_proposal_prob(
            x, y, sample, proposed_sample, step_size=step_size
        )
        q_star_to_q = self.compute_log_langevin_proposal_prob(
            x, y, proposed_sample, sample, step_size=step_size
        )
        return min(torch.tensor(0.0), u_q_star - u_q + q_star_to_q - q_to_q_star)

    def langevin_monte_carlo(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        simulation_steps: int = 1000,
        step_size: float = 1e-4,
        pbar: bool = True,
        metropolis_adjust: bool = True,  # MALA or ULA
    ):
        sample_dimension = self.layer_1.w.numel() + self.layer_2.w.numel()
        posterior_samples = torch.zeros((simulation_steps, sample_dimension))
        acceptance_counter = 0

        iter_pbar = tqdm(range(simulation_steps), disable=not pbar)

        for step in iter_pbar:
            metrics = defaultdict(float)
            current_sample = self.params_to_sample()
            proposed_sample = self.get_langevin_proposal(x, y, step_size=step_size)
            if metropolis_adjust:
                log_alpha = self.compute_log_lmc_acceptance(
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
            else:
                posterior_samples[step] = proposed_sample
                self.sample_to_params(proposed_sample)
                acceptance_counter += 1

            with torch.no_grad():
                metrics["log potential"] = float(
                    self.get_potential(x, y, self.params_to_sample())
                )
                metrics["average acceptance"] = acceptance_counter / (step + 1)
            iter_pbar.set_postfix(metrics)

        average_acceptance = acceptance_counter / simulation_steps

        return posterior_samples, average_acceptance


class HamiltonianPerceptron(MCMCPerceptron):
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

    def execute_leapfrog_step(
        self,
        sample: torch.Tensor,
        momentum: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        step_size: float = 1e-4,
    ) -> Tuple[torch.Tensor]:
        momentum_prime = momentum - (step_size / 2) * self.get_potential_grad(
            x, y, sample
        )
        new_sample = sample + step_size * momentum_prime
        new_momentum = momentum_prime - (step_size / 2) * self.get_potential_grad(
            x, y, new_sample
        )
        return new_sample, new_momentum

    def get_hamilton_proposal(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sample: torch.Tensor,
        momentum: torch.Tensor,
        step_size: float = 1e-4,
        leapfrog_steps: int = 50,
    ) -> Tuple[torch.Tensor]:

        new_sample = sample
        new_momentum = momentum

        for _ in range(leapfrog_steps):
            new_sample, new_momentum = self.execute_leapfrog_step(
                new_sample, new_momentum, x, y, step_size=step_size
            )

        return new_sample, new_momentum

    def compute_hamiltonian(
        self,
        sample: torch.Tensor,
        momentum: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        potential = self.get_potential(x, y, sample)
        kinetic = 0.5 * (torch.linalg.vector_norm(momentum) ** 2)
        return potential + kinetic

    def compute_log_hmc_acceptance(
        self,
        current_sample: torch.Tensor,
        current_momentum: torch.Tensor,
        proposal: torch.Tensor,
        proposal_momentum: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        proposal_hamiltonian = self.compute_hamiltonian(
            proposal, proposal_momentum, x, y
        )
        current_hamiltonian = self.compute_hamiltonian(
            current_sample, current_momentum, x, y
        )
        # if the leapfrog simulation is accurate enough,
        # the log acceptance probability should be zero due to energy conservation
        return min(torch.tensor(0.0), current_hamiltonian - proposal_hamiltonian)

    def hamiltonian_monte_carlo(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        simulation_steps: int = 1000,
        step_size: float = 1e-4,
        leapfrog_steps: int = 50,
        pbar: bool = True,
        metropolis_adjust: bool = True,
    ):
        sample_dimension = self.layer_1.w.numel() + self.layer_2.w.numel()
        posterior_samples = torch.zeros((simulation_steps, sample_dimension))
        acceptance_counter = 0

        iter_pbar = tqdm(range(simulation_steps), disable=not pbar)
        for step in iter_pbar:
            metrics = defaultdict(float)
            current_sample = self.params_to_sample()
            current_momentum = torch.randn_like(current_sample)
            proposed_sample, proposed_momentum = self.get_hamilton_proposal(
                x,
                y,
                current_sample,
                current_momentum,
                step_size=step_size,
                leapfrog_steps=leapfrog_steps,
            )
            if metropolis_adjust:
                log_alpha = self.compute_log_hmc_acceptance(
                    current_sample,
                    current_momentum,
                    proposed_sample,
                    proposed_momentum,
                    x,
                    y,
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
            else:
                posterior_samples[step] = proposed_sample
                self.sample_to_params(proposed_sample)
                acceptance_counter += 1

            with torch.no_grad():
                metrics["log potential"] = float(
                    self.get_potential(x, y, self.params_to_sample())
                )
                metrics["average acceptance"] = acceptance_counter / (step + 1)
            iter_pbar.set_postfix(metrics)

        average_acceptance = acceptance_counter / simulation_steps

        return posterior_samples, average_acceptance, metrics


    # this function is mainly for debugging purposes
    def get_energy_evolution(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        step_size: float = 1e-4,
        leapfrog_steps: int = 50,
        pbar: bool = True,
    ):
        new_sample = self.params_to_sample()
        new_momentum = torch.randn_like(new_sample)
        energy_evolution = torch.zeros(
            leapfrog_steps,
        )

        for i in tqdm(range(leapfrog_steps), disable=not pbar):
            new_sample, new_momentum = self.execute_leapfrog_step(
                new_sample, new_momentum, x, y, step_size=step_size
            )
            energy_evolution[i] = self.compute_hamiltonian(
                new_sample,
                new_momentum,
                x,
                y,
            ).detach()

        return energy_evolution
