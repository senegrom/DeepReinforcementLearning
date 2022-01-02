# SELF PLAY
EPISODES = 30
MCTS_SIMS = 50
MEMORY_SIZE = 30000
TURNS_UNTIL_TAU0 = 10  # turn on which it starts playing deterministically
CPUCT = 1
EPSILON = 0.2
ALPHA = 0.8

# RETRAINING
TRAINING_SIZE = 30000
EPOCHS = 10
REG_CONST = 0.0001
LEARNING_RATE = 0.001

HIDDEN_CNN_LAYERS = [
    {'filters': 75, 'kernel_size': (4, 4)},
    {'filters': 75, 'kernel_size': (4, 4)},
    {'filters': 75, 'kernel_size': (4, 4)},
    {'filters': 75, 'kernel_size': (4, 4)},
    {'filters': 75, 'kernel_size': (4, 4)},
    {'filters': 75, 'kernel_size': (4, 4)}
]

# EVALUATION
EVAL_EPISODES = 20
SCORING_THRESHOLD = 1.3
