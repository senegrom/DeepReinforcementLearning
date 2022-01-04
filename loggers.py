from initialise import run_folder
from utils import setup_logger

# SET all LOGGER_DISABLED to True to disable logging
# WARNING: the mcts log file gets big quite quickly

LOGGER_DISABLED = {
    'main': True
    , 'memory': True
    , 'tourney': False
    , 'mcts': True
    , 'model': False}

log_folder = run_folder / 'logs'
logger_mcts = setup_logger('logger_mcts', log_folder / 'logger_mcts.log')
logger_mcts.disabled = LOGGER_DISABLED['mcts']

logger_main = setup_logger('logger_main', log_folder / 'logger_main.log')
logger_main.disabled = LOGGER_DISABLED['main']

logger_tourney = setup_logger('logger_tourney', log_folder / 'logger_tourney.log')
logger_tourney.disabled = LOGGER_DISABLED['tourney']

logger_memory = setup_logger('logger_memory', log_folder / 'logger_memory.log')
logger_memory.disabled = LOGGER_DISABLED['memory']

logger_model = setup_logger('logger_model', log_folder / 'logger_model.log')
logger_model.disabled = LOGGER_DISABLED['model']
