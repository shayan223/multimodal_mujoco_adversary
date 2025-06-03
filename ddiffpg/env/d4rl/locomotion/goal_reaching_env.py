import numpy as np


def disk_goal_sampler(np_random, goal_region_radius=10.):
  th = 2 * np.pi * np_random.uniform()
  radius = goal_region_radius * np_random.uniform()
  return radius * np.array([np.cos(th), np.sin(th)])

def constant_goal_sampler(np_random, location=10.0 * np.ones([2])):
  return location

class GoalReachingEnv(object):
  """General goal-reaching environment."""
  BASE_ENV = None  # Must be specified by child class.

  def __init__(self, goal_sampler, eval=False, reward_type='dense'):
    self._goal_sampler = goal_sampler
    self._goal = np.ones([2])
    self.target_goal = self._goal

    # This flag is used to make sure that when using this environment
    # for evaluation, that is no goals are appended to the state
    self.eval = eval

    # This is the reward type fed as input to the goal confitioned policy
    self.reward_type = reward_type

  def _get_obs(self):
    base_obs = self.BASE_ENV._get_obs(self)
    goal_direction = self._goal - self.get_xy()
    if not self.eval:
      obs = np.concatenate([base_obs, goal_direction])
      return obs
    else:
      return base_obs

  def check_goal(self):
      if isinstance(self.target_goal, list):
        for goal in range(len(self.target_goal)):
          if np.linalg.norm(self.get_xy() - self.target_goal[goal]) <= 0.5:
            if self.target_goal[goal] in [(-8, 8), (-8, 8)]:
              return True, goal + 1, 20
            return True, goal + 1, 10
      else:
        if np.linalg.norm(self.get_xy() - self.target_goal) <= 0.5:
          return True, 1, 10

      return False, 0, 0

  def step(self, a):
    self.BASE_ENV.step(self, a)
    check, goal_id, reward = self.check_goal()
    
    done = False
    # Terminate episode when we reach a goal
    if self.eval and check:
      done = True
    
    obs = self._get_obs()
    return obs, reward, done, {'success': goal_id}

  def reset_model(self):
    if self.target_goal is not None or self.eval:
      self._goal = self.target_goal
    else:
      self._goal = self._goal_sampler(self.np_random)
    
    return self.BASE_ENV.reset_model(self)
