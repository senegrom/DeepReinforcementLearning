import random
from logging import Logger
from threading import Lock
from typing import Optional

import numpy as np
from futures import ThreadPoolExecutor, wait

import config
from agent import Agent, User, AbstractAgent
from game import Game
from memory import Memory
from model import Residual_CNN


def play_matches_between_versions(env, player1version: int, player2version: int, n_episodes: int,
                                  logger, turns_until_tau0: int, goes_first=0):
    player1: AbstractAgent
    player2: AbstractAgent
    if player1version == -1:
        player1 = User('player1', env.state_size, env.action_size)
    else:
        player1_nn = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape, env.action_size,
                                  config.HIDDEN_CNN_LAYERS)

        if player1version > 0:
            player1_network = player1_nn.read(env.name, player1version)
            player1_nn.model.set_weights(player1_network.get_weights())
        player1 = Agent('player1', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, player1_nn,
                        player1version)

    if player2version == -1:
        player2 = User('player2', env.state_size, env.action_size)
    else:
        player2_nn = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape, env.action_size,
                                  config.HIDDEN_CNN_LAYERS)

        if player2version > 0:
            player2_network = player2_nn.read(env.name, player2version)
            player2_nn.model.set_weights(player2_network.get_weights())
        player2 = Agent('player2', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, player2_nn,
                        player2version)

    scores, memory, points, sp_scores = play_matches(player1, player2, n_episodes, logger, turns_until_tau0, None,
                                                     goes_first)

    return scores, memory, points, sp_scores


_log_lock = Lock()
_score_lock = Lock()


def _play_match(player1: AbstractAgent, player2: AbstractAgent, e: int, logger: Logger, turns_until_tau0: int,
                memory: Optional[Memory], goes_first: int, scores, sp_scores, points):
    with _log_lock:
        logger.info('====================')
        logger.info(f'EPISODE {e + 1}')
        logger.info('====================')

        print(str(e + 1) + ' ', end='')

        env = Game()

        done = 0
        turn = 0
        player1.mcts = None
        player2.mcts = None

        if goes_first == 0:
            player1_starts = random.randint(0, 1) * 2 - 1
        else:
            player1_starts = goes_first

        if player1_starts == 1:
            players = {1: {"agent": player1, "name": player1.name}, -1: {"agent": player2, "name": player2.name}}
            logger.info(player1.name + ' plays as X')
        else:
            players = {1: {"agent": player2, "name": player2.name}, -1: {"agent": player1, "name": player1.name}}
            logger.info(player2.name + ' plays as X')
            logger.info('==============================================')

        env.game_state.render(logger)

    while done == 0:
        turn += 1

        # Run the MCTS algo and return an action
        if turn < turns_until_tau0:
            action, pi, mcts_value, nn_value = players[env.game_state.player_turn]['agent'].act(env.game_state, 1)
        else:
            action, pi, mcts_value, nn_value = players[env.game_state.player_turn]['agent'].act(env.game_state, 0)

        if memory is not None:
            # Commit the move to memory
            memory.commit_stmemory(env.identities, env.game_state, pi)

        with _log_lock:
            logger.info('action: %d', action)
            for r in range(env.grid_shape[0]):
                logger.info(['----' if x == 0 else '{0:.2f}'.format(np.round(x, 2)) for x in
                             pi[env.grid_shape[1] * r: (env.grid_shape[1] * r + env.grid_shape[1])]])

            if mcts_value is not None and nn_value is not None:
                logger.info('MCTS perceived value for %s: %f', env.game_state.pieces[env.game_state.player_turn],
                            np.round(mcts_value, 2))
                logger.info('NN perceived value for %s: %f', env.game_state.pieces[env.game_state.player_turn],
                            np.round(nn_value, 2))
            else:
                logger.info("External move")
            logger.info('--------------')

            state, value, done, _ = env.step(action)

            env.game_state.render(logger)
            logger.info('====================')

    if memory is not None:
        # If the game is finished, assign the values correctly to the game moves
        for move in memory.stmemory:
            if move['playerTurn'] == state.player_turn:
                move['value'] = value
            else:
                move['value'] = -value

        memory.commit_ltmemory()

    with _score_lock:
        with _log_lock:
            if value == 1:
                logger.info('%s WINS!', players[state.player_turn]['name'])
                scores[players[state.player_turn]['name']] += 1
                if state.player_turn == 1:
                    sp_scores['sp'] += 1
                else:
                    sp_scores['nsp'] += 1

            elif value == -1:
                logger.info('%s WINS!', players[-state.player_turn]['name'])
                scores[players[-state.player_turn]['name']] += 1

                if state.player_turn == 1:
                    sp_scores['nsp'] += 1
                else:
                    sp_scores['sp'] += 1

            else:
                logger.info('DRAW...')
                scores['drawn'] += 1
                sp_scores['drawn'] += 1

        pts = state.score
        points[players[state.player_turn]['name']].append(pts[0])
        points[players[-state.player_turn]['name']].append(pts[1])


PARALLEL = False


def play_matches(player1: AbstractAgent, player2: AbstractAgent, n_episodes: int, logger: Logger, turns_until_tau0: int,
                 memory: Optional[Memory] = None, goes_first: int = 0):
    scores = {player1.name: 0, "drawn": 0, player2.name: 0}
    sp_scores = {'sp': 0, "drawn": 0, 'nsp': 0}
    points = {player1.name: [], player2.name: []}

    if PARALLEL and isinstance(player1, Agent) and isinstance(player2, Agent):
        exe = ThreadPoolExecutor(50)
        tasks = []
    else:
        exe = None
        tasks = None

    for e in range(n_episodes):
        if isinstance(player1, Agent) and player1.version > 0:
            player1_nn = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, Game.input_shape, Game.action_size,
                                      config.HIDDEN_CNN_LAYERS)
            player1_network = player1_nn.read(Game.name, player1.version)
            player1_nn.model.set_weights(player1_network.get_weights())
            player1x = Agent(player1.name, Game.state_size, Game.action_size, config.MCTS_SIMS, config.CPUCT,
                             player1_nn, player1.version)
        else:
            player1x = player1

        if isinstance(player2, Agent) and player2.version > 0:
            player2_nn = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, Game.input_shape, Game.action_size,
                                      config.HIDDEN_CNN_LAYERS)
            player2_network = player2_nn.read(Game.name, player2.version)
            player2_nn.model.set_weights(player2_network.get_weights())
            player2x = Agent(player2.name, Game.state_size, Game.action_size, config.MCTS_SIMS, config.CPUCT,
                             player2_nn, player2.version)
        else:
            player2x = player2

        if tasks is not None:
            tasks.append(
                exe.submit(_play_match, player1x, player2x, e, logger, turns_until_tau0, memory, goes_first, scores,
                           sp_scores, points))
        else:
            _play_match(player1x, player2x, e, logger, turns_until_tau0, memory, goes_first, scores, sp_scores, points)

    if tasks is not None:
        wait(tasks)
        exe.shutdown()

    return scores, memory, points, sp_scores
