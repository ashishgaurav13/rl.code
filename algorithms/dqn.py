from utils import ExperienceBuffer
import keras
import numpy as np

class DQN:

	def __init__(self, env, buffer_size):
		# RL setup
		self.gamma = 0.99
		self.env = env
		# Experience buffer
		self.eb = ExperienceBuffer(buffer_size)
		# Construct DQN model
		obs_shape = env.observation_space.shape
		n_actions = env.action_space.n
		self.model = keras.models.Sequential()
		self.model.add(keras.layers.Dense(32, input_shape=obs_shape))
		self.model.add(keras.layers.Activation('relu'))
		self.model.add(keras.layers.Dense(32))
		self.model.add(keras.layers.Activation('relu'))
		self.model.add(keras.layers.Dense(n_actions))
		self.model.add(keras.layers.Activation('linear'))
		opt = keras.optimizers.Adam(lr=0.001)
		self.model.compile(loss='mse', optimizer=opt)
		# epsilon greedy setup
		self.eps = 1.0
		self.decay = 0.995

	def decay_eps(self):
		# Decay epsilon
		self.eps *= self.decay

	def choose_action(self, state):
		# Choose action randomly with probability eps, else
		# choose the greedy action
		if np.random.rand() <= self.eps:
			return np.random.choice(self.env.action_space.n)
		else:
			# This Q is for the current state. Choose the best
			# possible action, i.e. which has highest Q.
			[Q] = self.model.predict(np.array([state]))
			return np.argmax(Q)

	def train(self, batch_size):
		# Learn from batch_size random samples
		batch = self.eb.sample(batch_size)
		# Batch learning
		i, o = [], []
		for e in batch:
			state, action, r, next_state, done = e
			# To learn, the NN must have a target given by the Bellman target.
			# Bellman target is reward+max(all actions) {gamma.Q(next_state)}.
			# However, if the state is the last state of the episode (i.e. done
			# is true), then the target can be just reward.
			# Additionally Q(s, a) should try to learn the target. So, for
			# all other actions Q(s, ?), the target should just be what the
			# model predicts (i.e. no learning).
			[Q] = self.model.predict(np.array([state]))
			[Q_ns] = self.model.predict(np.array([next_state]))
			target = Q
			target[action] = r+self.gamma*np.max(Q_ns) if not done else r
			# Add to batch learning inputs and outputs
			i.append(state)
			o.append(target)
		# Keras fit
		return self.model.fit(np.array(i), np.array(o), verbose=0).\
			history['loss'][0] # just 1 item in history, for 1 epoch
