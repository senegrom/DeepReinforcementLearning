import numpy as np

import config


class Node:

    def __init__(self, state):
        self.state = state
        self.player_turn = state.player_turn
        self.id = state.id
        self.edges = []

    def is_leaf(self):
        if len(self.edges) > 0:
            return False
        else:
            return True


class Edge:
    def __init__(self, in_node, out_node, prior, action, turn=in_node.state.player_turn):
        self.id = in_node.state.id + '|' + out_node.state.id
        self.in_node = in_node
        self.out_node = out_node
        self.player_turn = turn
        self.action = action

        self.stats = {
            'N': 0,
            'W': 0,
            'Q': 0,
            'P': prior,
        }


class MCTS:

    def __init__(self, root, cpuct):
        self.root = root
        self.tree = {}
        self.cpuct = cpuct
        self.add_node(root)

    def __len__(self):
        return len(self.tree)

    def move_to_leaf(self):

        breadcrumbs = []
        current_node = self.root

        done = 0
        value = 0

        while not current_node.isLeaf():

            # lg.logger_mcts.info('PLAYER TURN...%d', current_node.state.playerTurn)

            max_qu = -99999

            if current_node is self.root:
                epsilon = config.EPSILON
                nu = np.random.dirichlet([config.ALPHA] * len(current_node.edges))
            else:
                epsilon = 0
                nu = [0] * len(current_node.edges)

            nb = 0
            for action, edge in current_node.edges:
                nb = nb + edge.stats['N']

            for idx, (action, edge) in enumerate(current_node.edges):

                U = self.cpuct * \
                    ((1 - epsilon) * edge.stats['P'] + epsilon * nu[idx]) * \
                    np.sqrt(nb) / (1 + edge.stats['N'])

                Q = edge.stats['Q']

                # lg.logger_mcts.info(
                #     'action: %d (%d)... N = %d, P = %f, nu = %f, adjP = %f, W = %f, Q = %f, U = %f, Q+U = %f'
                #     , action, action % 7, edge.stats['N'], np.round(edge.stats['P'], 6), np.round(nu[idx], 6),
                #     ((1 - epsilon) * edge.stats['P'] + epsilon * nu[idx])
                #     , np.round(edge.stats['W'], 6), np.round(Q, 6), np.round(U, 6), np.round(Q + U, 6))

                if Q + U > max_qu:
                    max_qu = Q + U
                    simulation_action = action
                    simulation_edge = edge

            new_state, value, done = current_node.state.take_action(simulation_action)
            current_node = simulation_edge.out_node
            breadcrumbs.append(simulation_edge)

        # lg.logger_mcts.info('DONE...%d', done)

        return current_node, value, done, breadcrumbs

    def back_fill(self, leaf, value, breadcrumbs):

        current_player = leaf.state.player_turn

        for edge in breadcrumbs:
            player_turn = edge.player_turn
            if player_turn == current_player:
                direction = 1
            else:
                direction = -1

            edge.stats['N'] = edge.stats['N'] + 1
            edge.stats['W'] = edge.stats['W'] + value * direction
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

    def add_node(self, node):
        self.tree[node.id] = node
