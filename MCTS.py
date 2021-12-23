from typing import Any, List, Tuple, Union, Optional

import numpy as np
from numpy import ndarray

import config
from abstractgame import AbstractGameState


class Node:

    def __init__(self, state: AbstractGameState) -> None:
        self.state = state
        self.player_turn = state.player_turn
        self.id = state.id
        self.edges = []

    def is_leaf(self) -> bool:
        if len(self.edges) > 0:
            return False
        else:
            return True


class Edge:
    def __init__(self, in_node: Node, out_node: Node, prior: float, action: int) -> None:
        self.id: str = in_node.state.id + '|' + out_node.state.id
        self.in_node: Node = in_node
        self.out_node: Node = out_node
        self.player_turn: int = in_node.state.player_turn
        self.action = action

        self.stats = {
            'N': 0,
            'W': 0.0,
            'Q': 0.0,
            'P': prior,
        }


class MCTS:

    def __init__(self, root: Node, cpuct: int) -> None:
        self.root = root
        self.tree = {}
        self.cpuct = cpuct
        self.add_node(root)

    def __len__(self):
        return len(self.tree)

    def move_to_leaf(self) -> Union[Tuple[Node, int, int, List[Any]], Tuple[Node, int, int, List[Edge]]]:

        breadcrumbs = []
        current_node = self.root

        done = 0
        value = 0

        while not current_node.is_leaf():

            max_qu = float('-inf')

            if current_node is self.root:
                epsilon = config.EPSILON
                nu = np.random.dirichlet([config.ALPHA] * len(current_node.edges))
            else:
                epsilon = 0
                nu = [0] * len(current_node.edges)

            nb = 0
            for action, edge in current_node.edges:
                nb = nb + edge.stats['N']

            simulation_action = None
            simulation_edge = None

            for idx, (action, edge) in enumerate(current_node.edges):

                u = self.cpuct * \
                    ((1 - epsilon) * edge.stats['P'] + epsilon * nu[idx]) * \
                    np.sqrt(nb) / (1 + edge.stats['N'])

                q = edge.stats['Q']

                if q + u > max_qu:
                    max_qu = q + u
                    simulation_action = action
                    simulation_edge: Optional[Edge] = edge

            assert simulation_action is not None and simulation_edge is not None, "No edges or better than -inf."

            new_state, value, done = current_node.state.take_action(simulation_action)
            current_node = simulation_edge.out_node
            breadcrumbs.append(simulation_edge)

        return current_node, value, done, breadcrumbs

    @staticmethod
    def back_fill(leaf: Node, value: Union[ndarray, int], breadcrumbs: List[Union[Any, Edge]]) -> None:

        current_player = leaf.state.player_turn

        for edge in breadcrumbs:
            player_turn = edge.player_turn
            if player_turn == current_player:
                direction = 1
            else:
                direction = -1

            edge.stats['N'] += 1
            edge.stats['W'] = edge.stats['W'] + value * direction
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

    def add_node(self, node: Node) -> None:
        self.tree[node.id] = node
