import gym
import numpy as np
import utils
from algorithms import DQN

## MountainCar-v0 has a per-step reward of -1. And the episodes can
## go for as long as 200 steps. This reward isn't actually well
## designed. An alternative strategy could be to give a 0 reward
## per step and give a +1 reward on success. In the following code,
## as you keep training, you should see the number of successful
## episodes per 100 episodes increase.

def episode(i, env, dqn, train): # i is the episode number
	# Reset
	obs = env.reset()
	# Statistics
	total_reward = 0
	total_loss = 0.0
	num_steps = 0
	while True:
		# Render the environment
		env.render()
		# Choose an action
		action = dqn.choose_action(obs)
		next_obs, r, done, info = env.step(action)
		num_steps += 1
		# Add to experience replay
		dqn.eb.add((obs, action, r, next_obs, done))
		# Train if needed, minibatch size 32
		if train: loss = dqn.train(32)
		# Decrease amount of exploration
		dqn.decay_eps()
		# Statistics
		total_reward += r
		if train: total_loss += loss
		# Episode terminated
		if done: break
	if train:
		print('%d: Episode Reward = %g, Avg Loss = %g' % (i,
			total_reward, total_loss/num_steps))
	else:
		print('%d: No training' % i)

# Create environment
env = gym.make('MountainCar-v0')
# DQN
dqn = DQN(env, buffer_size=15000)
# Explore and fill up some buffer
for ep_num in range(20): episode(ep_num+1, env, dqn, train=False)
# Do 200 episodes
for ep_num in range(700): episode(ep_num+1, env, dqn, train=True)
# Close the environment
utils.close(env)