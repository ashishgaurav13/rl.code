This repository contains implementations for some exercises of the reinforcement learning reading group. This edition happened every Monday 6:30pm-8pm (or Friday noon) and every week, we discussed a major topic in reinforcement learning. Initial topics were implementation driven and later topics became whiteboard discussions.

**Organized by**: [Data Science Club, University of Waterloo](http://waterloodatascience.club/)<br>
**Led by**: Ashish Gaurav, aka me

## Content

* **Week 1**: Basics [[slides]](http://bit.ly/DSCRL1)
	* _Topics_: MDPs, Monte Carlo Tree Search
	* _Paper_: [Mastering the game of Go with deep neural networks and tree search](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)
	* _Homework_: MCTS on TicTacToe [[environment]](https://github.com/ashishgaurav13/rl.code/blob/master/games/tic_tac_toe.py) [solutions: [mcts.py](https://github.com/ashishgaurav13/rl.code/blob/master/algorithms/mcts.py), [w1_mcts.py](https://github.com/ashishgaurav13/rl.code/blob/master/w1_mcts.py)]

* **Week 2**: Introductory Reinforcement Learning [[slides]](http://bit.ly/DSCRL2)
	* _Topics_: Policy Improvement Theorem, VI/PI, TD Methods, TD(0), Q Learning vs SARSA (off-policy vs on-policy)
	* _Paper_: [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
	* _Homework_: DQN on MountainCar-v0 [[environment]](https://gym.openai.com/envs/MountainCar-v0/) [solutions: [dqn.py](https://github.com/ashishgaurav13/rl.code/blob/master/algorithms/dqn.py), [w2_dqn.py](https://github.com/ashishgaurav13/rl.code/blob/master/w2_dqn.py)]
	
* **Week 3**: Exploration [[slides]](https://bit.ly/DSCRL3)
	* _Topics_: OFU, Count based exploration, IM
	* _Paper_: [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)

* **Week 4**: Policy Gradient
	* _Topics_: Deterministic Policy Gradient, DDPG
	* _Paper_: [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf)

* **Week 5**: TRPO
	* _Topics_: Derivation of TRPO surrogate objective, PPO (brief)
	* _Paper_: [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)

* **Week 6**: Meta Learning for RL
	* _Topics_: RL<sup>2</sup>, MAML
	* _Papers_: [RL<sup>2</sup>: Fast RL via Slow RL](https://arxiv.org/abs/1611.02779), [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)
