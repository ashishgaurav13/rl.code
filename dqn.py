import gym
import keras
import numpy as np
import utils

class ExperienceBuffer:

	def __init__(self, size):
		self.buffer = deque(maxlen=size)

	def add(self, experience):
		self.buffer.add(experience)

class DQN:

	def __init__(self, env):

		# Experience buffer
		self.eb = ExperienceBuffer(1500)

		# Construct DQN model
		state_dim = env.observation_space.shape
		n_actions = env.action_space.n
		self.model = keras.models.Sequential()
		self.model.add(keras.layers.Dense(32, input_dim=state_dim))
		self.model.add(keras.layers.Activation('relu'))
		self.model.add(keras.layers.Dense(32))
		self.model.add(keras.layers.Activation('relu'))
		self.model.add(keras.layers.Dense(n_actions))
		self.model.add(keras.layers.Activation('linear'))
		opt = keras.optimizers.Adam(lr=0.001)
		self.model.compile(loss='mse', optimizer=opt)
		
		# epsilon greedy setup
		self.eps = 1.0
		self.decay = 0.99

	def decay_eps(self):

		# Decay epsilon
		self.eps *= self.decay

	def learn(self, batch_size):

		# Learn from batch_size random samples
		batch = np.random.sample(self.eb, batch_size)
		for e in batch:
			next_state, 
			print(self.model.predict(next_state))

# Create environment
env = gym.make('MountainCar-v0')

# Reset
obs = env.reset()

while True:

	# Render the environment
	env.render()

	# Choose an action
	random_action = np.random.choice(env.action_space.n)
	next_obs, r, done, info = env.step(random_action)

	if done: break

# Close the environment
utils.close(env)