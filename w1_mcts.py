from games import TicTacToe
from algorithms import MCTS
import numpy as np
import tqdm

## 10000 searches per move is pretty strong. Vary the quality of
## decision making by changing number of searches from 10000 to
## something else. Or, try changing the board size and playing a game.

# Create environment
env = TicTacToe(3)
# Reset
obs = env.reset()
# whose turn? True if you want to play first
human_play = True
player = 'x'
other_player = 'o'
while True:
	# Print board and check if game over
	env.print_board()
	if env.won('x'):
		print('Game over, x wins')
		break
	if env.won('o'):
		print('Game over, o wins')
		break
	if env.draw():
		print('Draw')
		break
	if human_play:
		# ask for action
		i, j = map(int, input('Enter position:').split())
		action = (i, j)
	else:
		# MCTS
		mcts = MCTS(env, other_player)
		print('Thinking ...')
		for traversal_num in tqdm.tqdm(range(10000)): mcts.search(obs)
		# Ask for best action from obs
		# Uncertainty constant = 0
		mcts.reset()
		action = mcts.best_action(mcts.tree.root, 0)
	# step
	next_obs, r, _, _ = env.step(action, player=player if human_play \
		else other_player)
	obs = next_obs
	# next turn
	human_play = not human_play