from .gym_utils import *
from .experience_replay import *
import numpy as np

def random_choice(x):
	# Choose something from x
	random_idx = np.random.choice(len(x), 1)[0]
	return x[random_idx]