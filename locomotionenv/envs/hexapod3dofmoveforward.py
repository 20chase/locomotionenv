import os
import tempfile

import numpy as np

import gym
from gym.spaces import Tuple
from gym import utils
from gym.utils import EzPickle
from gym.envs.mujoco import mujoco_env




class Hexapod3DofMoveForwardEnv(mujoco_env.MujocoEnv, utils.EzPickle):
	"""docstring for ClassName"""

	def __init__(self):
		utils.EzPickle.__init__(self)
		xmlpath = os.path.join(os.path.dirname(__file__), "assets", "hexapod-3dof-moveforward.xml")
		mujoco_env.MujocoEnv.__init__(self, xmlpath,4)
		self.viewer = None


	def _step(self, a):
		
		xposbefore = self.get_body_com("torso")[0]
		self.do_simulation(a, self.frame_skip)
		xposafter = self.get_body_com("torso")[0]
		forward_reward = (xposafter - xposbefore)/self.dt

		ctrl_cost = 1e-1 * np.square(a).sum()

		contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.data.cfrc_ext, -1, 1)))

		agent_alive = self.get_body_com("torso")[2] >= 0.28
		survive_reward = 1.0 if agent_alive else -1


		reward = forward_reward - ctrl_cost - contact_cost + survive_reward
		state = self.state_vector()
		
		done = not agent_alive
		ob = self._get_obs()
		
		return ob, reward, done, dict(reward_forward=forward_reward,reward_ctrl=-ctrl_cost,reward_contact=-contact_cost,reward_survive=survive_reward)

	def _get_obs(self):
		return np.concatenate([
			self.data.qpos.flat,  # 躯干的坐标和四元数 7  其他关节的角度 3*6= 18
			self.data.qvel.flat[:6],  # 躯干的线速度 3 和 角速度 3  
			])

	def reset_model(self):
		qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
		qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
		self.set_state(qpos, qvel)
		return self._get_obs()

	
	def viewer_setup(self):
		self.viewer.cam.distance = self.model.stat.extent * 1.2
