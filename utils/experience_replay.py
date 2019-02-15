from collections import deque
import numpy as np

# A simple experience buffer which returns randomly sampled transitions
class ExperienceBuffer:

	def __init__(self, size):
		self.buffer = deque(maxlen=size)

	def add(self, experience):
		self.buffer.append(experience)

	def sample(self, batch_size):
		# Return a batch of batch_size samples, randomly chosen
		random_indices = np.random.choice(len(self.buffer), batch_size,
			replace=False)
		return np.array(self.buffer)[random_indices]

	def __len__(self):
		return len(self.buffer)