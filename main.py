# -*- coding: utf-8 -*-
# %matplotlib inline

import numpy as np

np.set_printoptions(suppress=True)

from shutil import copyfile
from importlib import reload

from game import Game
from agent import Agent
from memory import Memory
from model import Residual_CNN
from funcs import play_matches

import loggers as lg

from settings import run_folder, run_archive_folder
import initialise
import pickle
import config

lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')
lg.logger_main.info('=*=*=*=*=*=.      NEW LOG      =*=*=*=*=*')
lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')

env = Game()

## LOAD MEMORIES IF NECESSARY ##

if initialise.INITIAL_MEMORY_VERSION == None:
    memory = Memory()
else:
    print('LOADING MEMORY VERSION ' + str(initialise.INITIAL_MEMORY_VERSION) + '...')
    with open(f"{run_archive_folder}/{env.name}/memory/memory{initialise.INITIAL_MEMORY_VERSION:0>4}.p", "rb") as f:
        memory = pickle.load(f)

## LOAD MODEL IF NECESSARY ##

# create an untrained neural network objects from the config file
current_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (2,) + env.grid_shape, env.action_size,
                          config.HIDDEN_CNN_LAYERS)
best_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (2,) + env.grid_shape, env.action_size,
                       config.HIDDEN_CNN_LAYERS)

# If loading an existing neural netwrok, set the weights from that model
if initialise.INITIAL_MODEL_VERSION != None:
    best_player_version = initialise.INITIAL_MODEL_VERSION
    print('LOADING MODEL VERSION ' + str(initialise.INITIAL_MODEL_VERSION) + '...')
    m_tmp = best_NN.read(env.name, initialise.INITIAL_RUN_NUMBER, best_player_version)
    current_NN.model.set_weights(m_tmp.get_weights())
    best_NN.model.set_weights(m_tmp.get_weights())
# otherwise, just ensure the weights on the two players are the same
else:
    best_player_version = 0
    best_NN.model.set_weights(current_NN.model.get_weights())

# copy the config file to the run folder
copyfile('./config.py', run_folder + 'config.py')
# plot_model(current_NN.model, to_file=run_folder + 'models/model.png', show_shapes=True)

print('\n')

# CREATE THE PLAYERS ##

current_player = Agent('current_player', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, current_NN)
best_player = Agent('best_player', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, best_NN)
iteration = initialise.INITIAL_RUN_NUMBER if initialise.INITIAL_RUN_NUMBER is not None else 0

while 1:

    iteration += 1
    reload(lg)
    reload(config)

    print('ITERATION NUMBER ' + str(iteration))

    lg.logger_main.info('BEST PLAYER VERSION: %d', best_player_version)
    print('BEST PLAYER VERSION ' + str(best_player_version))

    ## SELF PLAY ##
    print('SELF PLAYING ' + str(config.EPISODES) + ' EPISODES...')
    _, memory, _, _ = play_matches(best_player, best_player, config.EPISODES, lg.logger_main,
                                   turns_until_tau0=config.TURNS_UNTIL_TAU0, memory=memory)
    print('\n')

    memory.clear_stmemory()

    if iteration % 5 == 0:
        pickle.dump(memory, open(run_folder + "memory/memory" + str(iteration).zfill(4) + ".p", "wb"))

    if len(memory.ltmemory) >= 1000:

        ## RETRAINING ##
        print('RETRAINING...')
        current_player.replay(memory.ltmemory)
        print('')

        ## TOURNAMENT ##
        print('TOURNAMENT...')
        scores, _, points, sp_scores = play_matches(best_player, current_player, config.EVAL_EPISODES,
                                                    lg.logger_tourney,
                                                    turns_until_tau0=0, memory=None)
        print('\nSCORES')
        print(scores)
        print('\nSTARTING PLAYER / NON-STARTING PLAYER SCORES')
        print(sp_scores)
        # print(points)

        print('\n\n')

        if scores['current_player'] > scores['best_player'] * config.SCORING_THRESHOLD:
            best_player_version = best_player_version + 1
            best_NN.model.set_weights(current_NN.model.get_weights())
            best_NN.write(env.name, best_player_version)

    else:
        print('MEMORY SIZE: ' + str(len(memory.ltmemory)))
