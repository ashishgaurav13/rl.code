def close(env):
	# Function to close environment, since otherwise it shows warnings
	# https://github.com/openai/gym/issues/893
	env.env.close()