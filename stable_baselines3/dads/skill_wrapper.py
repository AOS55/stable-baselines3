from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import gym
from gym import Wrapper


class SkillWrapper(Wrapper):

  def __init__(self,
               env: gym.Env,
               num_latent_skills: int = None,
               skill_type: str = 'discrete_uniform',
               preset_skill=None,
               min_steps_before_resample: int = 10,
               resample_prob: float = 0.):
    super(SkillWrapper, self).__init__(env)
    self._skill_type = skill_type
    if num_latent_skills is None:
      self._num_skills = 0
    else:
      self._num_skills = num_latent_skills
    self._preset_skill = preset_skill

    # attributes to control skill resampling
    self._min_steps_before_resample = min_steps_before_resample
    self._resample_prob = resample_prob

    if isinstance(self.env.observation_space, gym.spaces.Dict):
      size = self.env.observation_space.spaces['observation'].shape[0] + self._num_skills
    else:
      size = self.env.observation_space.shape[0] + self._num_skills
    self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype='float32')

  def _remake_time_step(self, cur_obs):
    if isinstance(self.env.observation_space, gym.spaces.Dict):
      cur_obs = cur_obs['observation']

    if self._num_skills == 0:
      return cur_obs
    else:
      return np.concatenate([cur_obs, self.skill])

  def _set_skill(self):
    if self._num_skills:
      if self._preset_skill is not None:
        self.skill = self._preset_skill
        print('Skill:', self.skill)
      elif self._skill_type == 'discrete_uniform':
        self.skill = np.random.multinomial(
          1, [1. / self._num_skills] * self._num_skills)
      elif self._skill_type == 'gaussian':
        self.skill = np.random.multivariate_normal(
          np.zeros(self._num_skills), np.eye(self._num_skills))
      elif self._skill_type == 'cont_uniform':
        self.skill = np.random.uniform(
          low=-1.0, high=1.0, size=self._num_skills)

  def reset(self):
    cur_obs = self.env.reset()
    self._set_skill()
    self._step_count = 0
    return self._remake_time_step(cur_obs)

  def step(self, action):
    cur_obs, reward, done, info = self.env.step(action)
    self._step_count += 1
    if self._preset_skill is None and self._step_count >= self._min_steps_before_resample and np.random.random(
    ) < self._resample_prob:
      self._set_skill()
      self._step_count = 0
    return self._remake_time_step(cur_obs), reward, done, info

  def close(self):
    return self.env.close()


if __name__ == '__main__':
  env = gym.make('CartPole-v1')
  wrapped_env = SkillWrapper(env, 4, 'discrete_uniform')
