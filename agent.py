# %matplotlib inline

import random
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import tensorflow as tf

import config
from MCTS import MCTS, Node, Edge
from abstractgame import AbstractGameState
from game import GameState
from model import Residual_CNN


class AbstractAgent(ABC):
    @abstractmethod
    def act(self, state, tau):
        pass

    def __init__(self, name: str) -> None:
        self.name: str = name


class User(AbstractAgent):
    def __init__(self, name, state_size, action_size):
        super().__init__(name)
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state: GameState, tau):
        print("\n")
        print(state.render(None))
        action = int(input('Enter your chosen action: '))
        pi = np.zeros(self.action_size)
        pi[action] = 1
        value = None
        nn_value = None
        return action, pi, value, nn_value


class Agent(AbstractAgent):
    def __init__(self, name: str, state_size: int, action_size: int, mcts_simulations: int, cpuct: int,
                 model: Residual_CNN, version: int) -> None:
        super().__init__(name)
        self.state_size = state_size
        self.action_size = action_size

        self.cpuct = cpuct

        self.mct_ssimulations = mcts_simulations
        self.model = model

        self.mcts = None

        self.train_overall_loss = []
        self.train_value_loss = []
        self.train_policy_loss = []
        self.val_overall_loss = []
        self.val_value_loss = []
        self.val_policy_loss = []
        self.version = version

    def simulate(self) -> None:
        # MOVE THE LEAF NODE
        leaf, value, done, breadcrumbs = self.mcts.move_to_leaf()

        # EVALUATE THE LEAF NODE
        value = self.evaluate_leaf(leaf, value, done)

        # BACKFILL THE VALUE THROUGH THE TREE
        self.mcts.back_fill(leaf, value, breadcrumbs)

    def act(self, state: GameState, tau: int) -> Tuple[int, np.ndarray, float, np.ndarray]:

        if self.mcts is None or state.id not in self.mcts.tree:
            self.build_mcts(state)
        else:
            self.change_root_mcts(state)

        # run the simulation
        for _ in range(self.mct_ssimulations):
            self.simulate()

        # get action values
        pi, values = self.get_av(1)

        # pick the action
        action, value = self.choose_action(pi, values, tau)
        next_state, _, _ = state.take_action(action)
        nn_value = -self.get_preds(next_state)[0]

        return action, pi, value, nn_value

    def get_preds(self, state: AbstractGameState) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        # predict the leaf

        preds = self.model.predict(tf.expand_dims(self.model.convert_to_model_input(state), 0))
        value_array = preds[0]
        logits_array = preds[1]
        value = value_array[0]

        logits = logits_array[0]

        allowed_actions = state.allowed_actions

        mask = np.ones(logits.shape, dtype=bool)
        mask[allowed_actions] = False
        logits[mask] = -100

        # SOFTMAX
        odds = np.exp(logits)
        probs = odds / np.sum(odds)

        return value, probs, allowed_actions

    def evaluate_leaf(self, leaf: Node, value: int, done: int) -> np.ndarray:
        if done == 0:
            value, probs, allowed_actions = self.get_preds(leaf.state)

            probs = probs[allowed_actions]

            for idx, action in enumerate(allowed_actions):
                new_state, _, _ = leaf.state.take_action(action)
                if new_state.id not in self.mcts.tree:
                    node = Node(new_state)
                    self.mcts.add_node(node)
                else:
                    node = self.mcts.tree[new_state.id]

                new_edge = Edge(leaf, node, probs[idx], action)
                leaf.edges.append((action, new_edge))

        return value

    def get_av(self, tau: int) -> Tuple[np.ndarray, np.ndarray]:
        edges = self.mcts.root.edges
        pi = np.zeros(self.action_size, dtype=np.integer)
        values = np.zeros(self.action_size, dtype=np.float32)

        for action, edge in edges:
            pi[action] = pow(edge.stats['N'], 1 / tau)
            values[action] = edge.stats['Q']

        pi = pi / (np.sum(pi) * 1.0)
        return pi, values

    @staticmethod
    def choose_action(pi: np.ndarray, values: np.ndarray, tau: int) -> Tuple[int, float]:
        if tau == 0:
            actions = np.argwhere(pi == max(pi))
            action = random.choice(actions)[0]
        else:
            action_idx = np.random.multinomial(1, pi)
            action = np.where(action_idx == 1)[0][0]

        value = values[action]

        return action, value

    def replay(self, ltmemory):
        for _ in range(config.TRAINING_LOOPS):
            minibatch = random.sample(ltmemory, min(config.BATCH_SIZE, len(ltmemory)))

            training_states = np.array([self.model.convert_to_model_input(row['state']) for row in minibatch])
            training_targets = {'value_head': np.array([row['value'] for row in minibatch]),
                                'policy_head': np.array([row['AV'] for row in minibatch])}

            fit = self.model.fit(training_states, training_targets, epochs=config.EPOCHS, verbose=1, validation_split=0,
                                 batch_size=32)

            self.train_overall_loss.append(round(fit.history['loss'][config.EPOCHS - 1], 4))
            self.train_value_loss.append(round(fit.history['value_head_loss'][config.EPOCHS - 1], 4))
            self.train_policy_loss.append(round(fit.history['policy_head_loss'][config.EPOCHS - 1], 4))

        print('\n')
        self.model.print_weight_averages()

    def predict(self, input_to_model):
        preds = self.model.predict(input_to_model)
        return preds

    def build_mcts(self, state: GameState) -> None:
        self.mcts = MCTS(Node(state), self.cpuct)

    def change_root_mcts(self, state: GameState) -> None:
        self.mcts.root = self.mcts.tree[state.id]
