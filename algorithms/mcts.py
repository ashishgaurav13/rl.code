import numpy as np
from utils import random_choice
from copy import deepcopy

class Node:

    def __init__(self, node_num):
        # node id
        self.num = node_num
        # state of the board corresponding to this node
        self.state = ''
        # adjacency list for this node
        self.edges = {}
        # parent id, useful for backup
        self.parent_num = None
        # node properties
        self.N = 0
        self.Q = 0

    def is_terminal(self):
        return self.edges == {}

class Tree:

    def __init__(self):
        # list of nodes
        self.nodes = {}
        # When making a new node, use this node id, and
        # increment this node id, so the next node has
        # a different node id.
        self.new_node_num = 0
        # Root node
        self.root = self._create_node()
        # Which node are we currently on? For traversal
        self.curr_node_num = 0
        self.latest_obs = None

    def _create_node(self):
        # Internal function to create a node, without edges
        created_node = Node(self.new_node_num)
        self.nodes[self.new_node_num] = created_node
        self.new_node_num += 1
        return created_node

    def new_node(self, action):
        # Create a new node from current node, after taking action
        created_node = self._create_node()
        self.nodes[self.curr_node_num].edges[action] = created_node.num
        created_node.parent_num = self.curr_node_num

    def move(self, action):
        # Move to an existing node from current node, taking action
        possible_edges = self.nodes[self.curr_node_num].edges
        assert(action in possible_edges.keys())
        self.curr_node_num = possible_edges[action]

    def add_state(self, obs):
        # Put the latest observation into the latest node
        if self.nodes[self.curr_node_num].state == '':
            self.nodes[self.curr_node_num].state = obs
        self.latest_obs = obs

class MCTS:

    def __init__(self, env, player='x'):
        # We shouldn't modify the environment, so we do traversals
        # on copies of the environment
        self.original_env = env
        self.tree = Tree()
        self.player = player
        self.other_player = 'x' if player == 'o' else 'o'
        # Is it our turn? In the beginning, yes
        self.turn = True

    def reset(self):
        # Sets current node to root.
        self.tree.curr_node_num = 0
        # Our turn
        self.turn = True
        # Create an env which is the deepcopy of original_env
        self.env = deepcopy(self.original_env)

    def _get_possible_actions(self):
        # Return all possible actions from current node
        n = self.env.n
        free_positions = [(i, j) for i in range(n) for j in range(n)\
            if self.env.board[i][j] is None]
        return set(free_positions)

    def move(self, action):
        # Move in the MCTS tree. This means moving in the tree, updating
        # state information and taking the action.
        if self.turn:
            next_obs, R, _, _ = self.env.step(action, self.player)
        else:
            next_obs, R, _, _ = self.env.step(action, self.other_player)
        self.tree.move(action)
        self.tree.add_state(next_obs)
        return R

    def search(self, obs):
        # Perform a traversal from the root node.
        self.reset() # reset
        self.tree.add_state(obs)
        # Reach a leaf following tree policy
        self.tree_policy()
        # If we terminated, backup win/loss
        if self.env.won(self.player):
            self.backup(1.0)
            return
        if self.env.won(self.other_player):
            self.backup(-1.0)
            return
        if self.env.draw():
            self.backup(0.0)
            return
        # Perform a rollout and back up
        rollout_reward = self.default_policy() # from leaf node
        self.backup(rollout_reward)

    def tree_policy(self):
        # Policy that determines how to move through the MCTS tree.
        while not (self.env.won('x') or self.env.won('o') or self.env.draw()):
            # Get current node
            node = self.tree.nodes[self.tree.curr_node_num]
            # Get possible actions
            possible_actions = self._get_possible_actions()
            # Get already performed actions
            already_done = set(node.edges.keys())
            # What have we not tried?
            not_tried = possible_actions - already_done
            # Expand if it is possible to perform a new action
            if len(not_tried) > 0:
                self.expand(node, not_tried)
                self.turn = not self.turn
                return
            else:
                if self.turn:
                    # If our turn, choose what is suggested through UCB
                    action = self.best_action(node, 1)
                else:
                    action = random_choice(list(possible_actions))
                self.move(action)
                self.turn = not self.turn

    def expand(self, node, not_tried):
        # Create a new node from the given node. Chooses an action from the
        # not_tried set. Also moves to the newly created node.
        random_action = random_choice(list(not_tried))
        self.tree.new_node(random_action)
        return self.move(random_action)

    def best_action(self, node, C):
        # Find the best option to execute from a given node. The constant
        # C determines the coefficient of the uncertainty estimate.
        # Assume its our turn.
        assert(self.turn)
        next_actions = list(node.edges.keys())
        Q_UCB = {}
        for action in next_actions:
            next_node_num = node.edges[action]
            next_node = self.tree.nodes[next_node_num]
            Q_UCB[action] = 0
            # Q/N is the empirical estimate. Using N+1 instead of N so that
            # we don't get a 0 denominator
            Q_UCB[action] += next_node.Q / (next_node.N + 1)
            # Current board
            obs = self.tree.latest_obs
            # Uncertainty term
            Q_UCB[action] += C * np.sqrt(2.0*np.log(node.N)/(next_node.N+1))
        # Choose argmax
        k, v = list(Q_UCB.keys()), list(Q_UCB.values())
        # Uncomment the next two lines to see the predicted win probabilities
        # if C == 0:
        #     print(Q_UCB)
        return k[np.argmax(v)]

    def default_policy(self):
        # Default policy, used for rollouts.
        rollout_reward = 0
        obs = self.tree.latest_obs
        while not (self.env.won('x') or self.env.won('o') or self.env.draw()):
            # Choose a random action in the rollout.
            possible_actions = self._get_possible_actions()
            random_action = random_choice(list(possible_actions))
            if self.turn:
                next_obs, R, _, _ = self.env.step(random_action, self.player)
            else:
                next_obs, R, _, _ = self.env.step(random_action, \
                    self.other_player)
            obs = next_obs
            rollout_reward += R
            self.turn = not self.turn
        return rollout_reward

    def backup(self, rollout_reward):
        # Reward backup strategy.
        curr_node = self.tree.nodes[self.tree.curr_node_num]
        while curr_node is not None:
            curr_node.N += 1
            curr_node.Q += rollout_reward
            if curr_node.parent_num is not None:
                curr_node = self.tree.nodes[curr_node.parent_num]
            else:
                curr_node = None
