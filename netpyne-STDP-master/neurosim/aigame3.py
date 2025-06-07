import brax
from brax import envs
from brax.envs import State
import numpy as np
from datetime import datetime
from collections import deque
import random
import jax
import jax.numpy as jnp
from gym import spaces

class AIGame:
  """ Interface to Brax game (replacing OpenAI Gym)
  """

  def __init__(self, config):
    self.conf_env = config['env']
    self.do_render = self.conf_env['render']
    self.actionsPerPlay = config['actionsPerPlay']
    self.observations = deque(maxlen=config['observationsToKeep'])
    self.rewards = deque(maxlen=config['rewardsToKeep'])
    self.count_episodes = 0
    self.count_steps = [0]
    self.tstart = None
    self._setup_env()
    self.state = self.env.reset(rng=jax.random.PRNGKey(0))
    obs_adjusted = np.array([self.state.obs[0], self.state.obs[2], self.state.obs[1], self.state.obs[3]])
    self.observations.append(obs_adjusted)

    # 检查动作相关参数

  def _setup_env(self):
    env_name = "inverted_pendulum"
    env = envs.create(env_name=env_name)

    if 'seed' in self.conf_env:
      self.rng = jax.random.PRNGKey(self.conf_env['seed'])
    else:
      self.rng = jax.random.PRNGKey(0)
    if 'rerunEpisode' in self.conf_env:
      for _ in range(self.conf_env['rerunEpisode'] - 1):
        self.rng, key = jax.random.split(self.rng)
        env.reset(key)
    self.env = env
    # 设置与Gym CartPole一致的observation_space（仅用于外部访问）
    self.env.observation_space = spaces.Box(
      low=np.array([-4.8, -np.inf, -0.418, -np.inf]),  # [x, dx/dt, theta, dtheta/dt]
      high=np.array([4.8, np.inf, 0.418, np.inf]),
      dtype=np.float32
    )
    # Gym的实际终止边界
    self.x_threshold = 2.4
    self.theta_threshold = 0.2095  # 12度，约0.2095弧度

  def _clean(self):
    self.observations.clear()
    self.rewards.clear()
    self.count_steps.append(0)
    self.tstart = None
    if 'rerunEpisode' in self.conf_env:
      self._setup_env()
    self.rng, key = jax.random.split(self.rng)
    self.state = self.env.reset(rng=key)
    obs_adjusted = np.array([self.state.obs[0], self.state.obs[2], self.state.obs[1], self.state.obs[3]])
    self.observations.append(obs_adjusted)

  def randmove(self):
    if 'rerunEpisode' in self.conf_env:
      return random.randint(0, 1)
    self.rng, key = jax.random.split(self.rng)
    action = jax.random.uniform(key, shape=(1,), minval=-1.0, maxval=1.0)
    return 1 if action > 0 else 0

  def playGame(self, actions):
    current_rewards = []
    done = False
    if not self.tstart:
        self.tstart = datetime.now()

    assert len(actions) == self.actionsPerPlay
    for adx in range(self.actionsPerPlay):
        caction = actions[adx]
        # 将Gym的0/1映射到Brax的[-10, 10]，匹配Gym的10N力
        caction = -0.1 if caction == 0 else 0.1
        caction = jnp.array([caction], dtype=jnp.float32)

        self.state = self.env.step(self.state, caction)
        observation = np.array([self.state.obs[0], self.state.obs[2], self.state.obs[1], self.state.obs[3]])
        reward = 1.0
        x = observation[0]
        theta = observation[2]
        done = (x < -self.x_threshold or x > self.x_threshold or
                theta < -self.theta_threshold or theta > self.theta_threshold)
        #print("Position:",x,"Angle",theta,"Acton:",caction,actions)

        done = done or (self.state.done > 0.5)

        if self.do_render:
            print("Rendering not fully implemented; use brax.io.html for visualization.")

        current_rewards.append(reward)
        self.rewards.append(reward)
        self.observations.append(observation)

        self.count_steps[-1] += 1

        if done:
            self._clean()
            self.count_episodes += 1
            break

    return current_rewards, done