import click
import gym
import os

import numpy as np
import tensorflow as tf

import locomotionenv.envs 



def main():
	env = gym.make("Hexapod-3Dof-MoveForward-v1")
	#env = gym.make("Hexapod-2Dof-MoveForward-v1")
	print(env.env.observation_space)
	print(env.env.action_space)
	for i_episode in range(20):
		observation = env.reset()
		for t in range(10000):
			env.render()
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
			if done:
				print("Episode finished after {} timesteps".format(t+1))
				break


if __name__ == "__main__":
    main()