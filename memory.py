from collections import deque

import config
from abstractgame import AbstractGameState


class Memory:
    def __init__(self):
        self.MEMORY_SIZE = config.MEMORY_SIZE
        self.ltmemory = deque(maxlen=config.MEMORY_SIZE)
        self.stmemory = deque(maxlen=config.MEMORY_SIZE)

    def commit_stmemory(self, identities, state: AbstractGameState, action_values):
        Memory.commit_to(identities, state, action_values, self.stmemory) #todo for parallel

    @staticmethod
    def commit_to(identities, state: AbstractGameState, action_values, stmemory: deque):
        for r in identities(state, action_values):
            stmemory.append({
                'board': r[0].board,
                'state': r[0],
                'id': r[0].id,
                'AV': r[1],
                'playerTurn': r[0].player_turn
            })

    def commit_ltmemory(self):
        for i in self.stmemory:
            self.ltmemory.append(i)
        self.clear_stmemory()

    def clear_stmemory(self):
        self.stmemory = deque(maxlen=config.MEMORY_SIZE)
