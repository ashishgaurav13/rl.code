import numpy as np

class TicTacToe:

	def __init__(self, n):
		self.n = n

	def reset(self):
		self.board = [[None for i in range(self.n)] for j in range(self.n)]
		# return initial state
		return self.board

	def step(self, action):
		# ensure action is valid (i, j)
		assert(len(action) == 2)
		assert(type(action[0]) == int and type(action[1]) == int)
		assert(0 <= action[0] < self.n)
		assert(0 <= action[1] < self.n)
		i, j = action
		assert(self.board[i][j] == None)
		# do the step
		self.board[i][j] = 'x'
		# ask ai to play next move
		self.ai()
		# if game over, return +1 if win, -1 if lose
		# if game not over, return 0
		if self.won('x'):
			return self.board, 1, True, {}
		elif self.won('o'):
			return self.board, -1, True, {}
		else:
			return self.board, 0, False, {}

	def won(self, player):
		# check rows
		for i in range(self.n):
			if self.board[i].count(player) == self.n:
				return True
		# check cols
		board_transpose = [*zip(*self.board)]
		for i in range(self.n):
			if board_transpose[i].count(player) == self.n:
				return True
		# check diag
		left_diag, right_diag = True, True
		for i in range(self.n):
			if self.board[i][i] != player:
				left_diag = False
			if self.board[i][-i-1] != player:
				right_diag = False
		return left_diag or right_diag

	def print_board(self):
		for i in range(self.n):
			for j in range(self.n):
				if self.board[i][j] == None:
					print('[ ]', end='')
				elif self.board[i][j] == 'x':
					print('[x]', end='')
				else:
					print('[o]', end='')
			print('')

	def ai(self):
		# randomly choose a play
		free_positions = [(i, j) for i in range(self.n) for j in range(self.n)\
			if self.board[i][j] is None]
		i, j = free_positions[np.random.choice(len(free_positions))]
		self.board[i][j] = 'o'

	def random_game(self):
		init_obs = self.reset()
		while True:
			self.print_board()
			if self.won('x'):
				print('Game over, x wins')
				break
			elif self.won('o'):
				print('Game over, o wins')
				break
			i, j = map(int, input('Enter position:').split())
			self.step((i, j))