from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.sac.sac import SAC
from stable_baselines3.dads.skill_dynamics import SkillDynamics


class DADS(SAC):

  def __init__(self,
               skill_dynamics_observation_size,
               observation_modify_fn=None,
               restrict_input_size: int = 0,
               latent_size: int = 2,
               latent_prior: str = 'cont_uniform',
               prior_samples: int = 100,
               fc_layer_params: tuple = (256, 256),
               normalize_observations: bool = True,
               network_type: str = 'default',
               num_mixture_components: int = 4,
               fix_variance: bool = True,
               skill_dynamics_learning_rate: float = 3e-4,
               reweigh_batches: bool = False,
               *sac_args,
               **sac_kwargs):
    super(DADS, self).__init__(*sac_args, **sac_kwargs)

    self._skill_dynamics_learning_rate = skill_dynamics_learning_rate
    self._latent_size = latent_size
    self._latent_prior = latent_prior
    self._prior_samples = prior_samples
    self._restrict_input_size = restrict_input_size
    self._process_observation = observation_modify_fn

    self._skill_dynamics = SkillDynamics(observation_size=skill_dynamics_observation_size,
                                         action_size=latent_size,
                                         restrict_observation=self._restrict_input_size,
                                         normalize_observations=normalize_observations,
                                         fc_layer_params=fc_layer_params,
                                         network_type=network_type,
                                         num_components=num_mixture_components,
                                         fix_variance=fix_variance,
                                         reweigh_batches=reweigh_batches
                                         )

  def train(self, gradient_steps: int, batch_size: int = 64) -> None:
    self.policy.set_training_mode(True)
    optimizers = [self.actor.optimizer, self.critic.optimizer]
    if self.ent_coef_optimizer is not None:
      optimizers += [self.ent_coef_optimizer]

    self._update_learning_rate(optimizers)

    ent_coef_losses, ent_coefs = [], []
    actor_losses, critic_losses = [], []

    for gradient_step in range(gradient_steps):
      replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

      if self.use_sde:
        self.actor.reset_noise()

      actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
      log_prob = log_prob.reshape(-1, 1)
      ent_coef_loss = None
      if self.ent_coef_optimizer is not None:
        ent_coef = th.exp(self.log_ent_coef.detach())
        ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
        ent_coef_losses.append(ent_coef_loss.item())
      else:
        ent_coef = self.ent_coef_tensor

      ent_coefs.append(ent_coef.item())

      # Optimize entropy coefficient, also called
      # entropy temperature or alpha in the paper
      if ent_coef_loss is not None:
        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()

      with th.no_grad():
        # Select action according to policy
        next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
        # Compute the next Q values: min over all critics targets
        next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
        next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
        # add entropy term
        next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
        # td error + entropy term if using extrinsic reward
        target_q_values_extrinsic = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

        target_q_values = self.compute_dads_rewards(replay_data.observations, cur_skill, target_obs) +\
                          (1 - replay_data.dones) * self.gamma * next_q_values

      # Get current Q-values estimates for each critic network
      # using action from the replay buffer
      current_q_values = self.critic(replay_data.observations, replay_data.actions)

      # Compute critic loss
      critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
      critic_losses.append(critic_loss.item())

      # Optimize the critic
      self.critic.optimizer.zero_grad()
      critic_loss.backward()
      self.critic.optimizer.step()

      # Compute actor loss
      # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
      # Mean over all critic networks
      q_values_pi = th.cat(self.critic.forward(replay_data.observations, actions_pi), dim=1)
      min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
      actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
      actor_losses.append(actor_loss.item())

      # Optimize the actor
      self.actor.optimizer.zero_grad()
      actor_loss.backward()
      self.actor.optimizer.step()

      # Update target networks
      if gradient_step % self.target_update_interval == 0:
        polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

    self._n_updates += gradient_steps

    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    self.logger.record("train/ent_coef", np.mean(ent_coefs))
    self.logger.record("train/actor_loss", np.mean(actor_losses))
    self.logger.record("train/critic_loss", np.mean(critic_losses))
    if len(ent_coef_losses) > 0:
      self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

  def compute_dads_rewards(self, input_obs, cur_skill, target_obs):
    if self._process_observation is not None:
      input_obs, target_obs = self._process_observation(input_obs), self._process_observation(target_obs)

    num_reps = self._prior_samples if self._prior_samples > 0 else self._latent_size - 1
    input_obs_altz = np.concatenate([input_obs] * num_reps, axis=1)
    target_obs_altz = np.concatenate([target_obs] * num_reps, axis=0)

    # for marginalization of the denominator
    if self._latent_prior == 'discrete_uniform' and not self._prior_samples:
      alt_skill = np.concatenate(
        [np.roll(cur_skill, idx, axis=1) for idx in range(1, num_reps + 1)],
        axis=0
      )
    elif self._latent_prior == 'discrete_uniform':
      alt_skill = np.random.multinomial(
        1, [1. / self._latent_size] * self._latent_size,
        size=input_obs_altz.shape[0]
      )
    elif self._latent_prior == 'gaussian':
      alt_skill = np.random.multivariate_normal(
        np.zeros(self._latent_size),
        np.eye(self._latent_size),
        size=input_obs_altz.shape[0]
      )
    elif self._latent_prior == 'cont_uniform':
      alt_skill = np.random.uniform(
        low=-1.0, high=1.0, size=(input_obs_altz.shape[0], self._latent_size)
      )

    logp = self._skill_dynamics.get_log_prob(input_obs, cur_skill, target_obs)

    split_group = 20*4000
    if input_obs_altz.shape[0] <= split_group:
      logp_altz = self._skill_dynamics.get_log_prob(input_obs_altz, alt_skill, target_obs_altz)
    else:
      logp_altz = []
      for split_idx in range(input_obs_altz.shape[0] // split_group):
        start_split = split_idx * split_group
        end_split = (split_idx + 1) * split_group
        logp_altz.append(
          self._skill_dynamics.get_log_prob(
            input_obs_altz[start_split:end_split],
            alt_skill[start_split:end_split],
            target_obs_altz[start_split:end_split]
          )
        )
      if input_obs_altz.shape[0] % split_group:
        logp_altz.append(
          self._skill_dynamics.get_log_prob(input_obs_altz[-start_split:],
                                            alt_skill[-start_split:],
                                            target_obs_altz[-start_split:])
        )
      logp_altz = np.concatenate(logp_altz)
    logp_altz = np.array(np.array_split(logp_altz, num_reps))

    intrinsic_reward = np.log(num_reps + 1) - np.log(1 + np.exp(np.clip(logp_altz - logp.reshape(1, -1), -50, 50)
                                                                ).sum(axis=0))
    return intrinsic_reward, {'logp': logp, 'logp_altz': logp_altz.flatten()}
