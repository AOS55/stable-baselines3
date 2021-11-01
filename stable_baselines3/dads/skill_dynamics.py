from typing import Any, Dict, List, Optional, Tuple, Type, Union

import os
import torch as th
from torch import nn
import numpy as np


class SimpleDynamics(nn.Module):

  def __init__(self,
               observation_size: int,
               action_size: int,
               num_time_steps: int,
               fc_layer_params=(256, 256)):

    super(SimpleDynamics, self).__init__()
    self.dynamics_network = nn.Sequential()
    # self.mean_network = nn.Linear(fc_layer_params[-1], observation_size)
    # self.var_network = nn.Sequential()
    self.dynamics_network.add_module('input', nn.Linear(action_size, fc_layer_params[0]))
    self.dynamics_network.add_module('activation_' + 'head', nn.ReLU())
    self.dynamics_network.add_module('hidden', nn.Linear(fc_layer_params[0], fc_layer_params[-1]))
    self.dynamics_network.add_module('activation_' + 'hidden', nn.ReLU())
    # self.var_network.add_module('stddev', nn.Linear(fc_layer_params[-1], observation_size))
    # self.var_network.add_module('softmax', nn.Softplus())

  def forward(self,
              actions: th.tensor):
    observations = self.dynamics_network(actions)
    # mean = self.mean_network(observations)
    # var = self.var_network(observations)
    return observations


class SkillDynamics:

  def __init__(
      self,
      observation_size: int,
      action_size: int,
      restrict_observation: int = 0,
      normalize_observations: bool = False,
      fc_layer_params=(256, 256),
      network_type: str = 'default',
      num_components: int = 1,
      fix_variance: bool = False,
      reweigh_batches: bool = False
    ):
    self._observation_size = observation_size
    self._action_size = action_size
    self._normalize_observations = normalize_observations
    self._restrict_observation = restrict_observation
    self._reweigh_batches = reweigh_batches

    # dynamics-network properties
    self._fc_layer_params = fc_layer_params
    self._num_componenets = num_components
    self._fix_variance = fix_variance
    if not self._fix_variance:
      self._std_lower_clip = 0.3
      self._std_upper_clip = 10.0
    self.dynamics_network = nn.Sequential()
    if network_type == 'default':
      for idx, layer_size in enumerate(fc_layer_params):
        self.dynamics_network.add_module('hid_' + str(idx), nn.Linear(self._observation_size, layer_size))
        self.dynamics_network.add_module('activation_' + str(idx), nn.ReLU())

    self._use_placeholders = False
    self.log_probability = None
    self.dyn_max_op = None
    self.dyn_min_op = None
    self._use_modal_mean = False

    self._saver = None

  def _network_with_separate_skills(self, timesteps, actions):

    skill_out = actions
    skill_network = nn.Sequential()
    for idx, layer_size in enumerate((self._fc_layer_params[0] // 2,)):
      skill_network.add_module('hid_' + str(idx), nn.Linear(skill_out, layer_size))
      skill_network.add_module('activation_' + str(idx), nn.ReLU())

    ts_out = timesteps
    ts_network = nn.Sequential()
    for idx, layer_size in enumerate((self._fc_layer_params[0] // 2,)):
      ts_network.add_module('hid_' + str(idx), nn.Linear(ts_out, layer_size))
      skill_network.add_module('activation_' + str(idx), nn.ReLU())

    out = th.cat((ts_out, skill_out), dim=1)
    out_network = nn.Sequential()
    for idx, layer_size in enumerate(self._fc_layer_params[1:]):
      out_network.add_module('hid_' + str(idx), nn.Linear(out.shape, layer_size))
      out_network.add_module('activation_' + str(idx), nn.ReLU())

    return self._get_distribution(out)

  def _get_distribution(self, out):
    if self._num_componenets > 1:
      logit_network = nn.Sequential()
      logit_network.add_module('logits', nn.Linear(out, self._num_componenets))
      means, scale_diags = [], []
      for component_id in range(self._num_componenets):
        means.append(

        )

  def get_log_prob(self, input_obs, cur_skill, target_obs):
    # get the log prob by sampling the current skill with target obs and input obs
    logp = []
    return logp


if __name__ == '__main__':
  obs_size = 30
  act_size = 4
  steps = 250
  action_tensor = th.rand(act_size)
  simple_dyn_network = SimpleDynamics(obs_size, act_size, steps)
  observations = simple_dyn_network(action_tensor)
  pred_prob = nn.Softplus()(observations)
  y_pred = pred_prob.argmax(0)
  print(y_pred)
