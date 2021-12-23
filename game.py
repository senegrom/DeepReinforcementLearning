from logging import Logger
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from numpy import int64, ndarray

from abstractgame import AbstractGame, AbstractGameState


# noinspection DuplicatedCode
class GameState(AbstractGameState):
    winners = [
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [7, 8, 9, 10],
        [8, 9, 10, 11],
        [9, 10, 11, 12],
        [10, 11, 12, 13],
        [14, 15, 16, 17],
        [15, 16, 17, 18],
        [16, 17, 18, 19],
        [17, 18, 19, 20],
        [21, 22, 23, 24],
        [22, 23, 24, 25],
        [23, 24, 25, 26],
        [24, 25, 26, 27],
        [28, 29, 30, 31],
        [29, 30, 31, 32],
        [30, 31, 32, 33],
        [31, 32, 33, 34],
        [35, 36, 37, 38],
        [36, 37, 38, 39],
        [37, 38, 39, 40],
        [38, 39, 40, 41],

        [0, 7, 14, 21],
        [7, 14, 21, 28],
        [14, 21, 28, 35],
        [1, 8, 15, 22],
        [8, 15, 22, 29],
        [15, 22, 29, 36],
        [2, 9, 16, 23],
        [9, 16, 23, 30],
        [16, 23, 30, 37],
        [3, 10, 17, 24],
        [10, 17, 24, 31],
        [17, 24, 31, 38],
        [4, 11, 18, 25],
        [11, 18, 25, 32],
        [18, 25, 32, 39],
        [5, 12, 19, 26],
        [12, 19, 26, 33],
        [19, 26, 33, 40],
        [6, 13, 20, 27],
        [13, 20, 27, 34],
        [20, 27, 34, 41],

        [3, 9, 15, 21],
        [4, 10, 16, 22],
        [10, 16, 22, 28],
        [5, 11, 17, 23],
        [11, 17, 23, 29],
        [17, 23, 29, 35],
        [6, 12, 18, 24],
        [12, 18, 24, 30],
        [18, 24, 30, 36],
        [13, 19, 25, 31],
        [19, 25, 31, 37],
        [20, 26, 32, 38],

        [3, 11, 19, 27],
        [2, 10, 18, 26],
        [10, 18, 26, 34],
        [1, 9, 17, 25],
        [9, 17, 25, 33],
        [17, 25, 33, 41],
        [0, 8, 16, 24],
        [8, 16, 24, 32],
        [16, 24, 32, 40],
        [7, 15, 23, 31],
        [15, 23, 31, 39],
        [14, 22, 30, 38],
    ]
    pieces = ["-", "X", "O"]  # -1 for O

    def __init__(self, board: ndarray, player_turn: int) -> None:
        super().__init__(board, player_turn)

    @property
    def id(self) -> str:
        return ''.join(map(str, self.binary))

    @property
    def allowed_actions(self) -> list:
        allowed = []
        for i in range(len(self.board)):
            if i >= len(self.board) - 7:
                if self.board[i] == 0:
                    allowed.append(i)
            else:
                if self.board[i] == 0 and self.board[i + 7] != 0:
                    allowed.append(i)

        return allowed

    @property
    def binary(self) -> tf.Tensor:
        position = tf.stack([
            tf.where(self.board == self.player_turn, 1, 0),
            tf.where(self.board == -self.player_turn, 1, 0)
        ], axis=0)

        return position

    def _convert_state_to_id(self):
        player1_position = np.zeros(len(self.board), dtype=np.int)
        player1_position[self.board == 1] = 1

        other_position = np.zeros(len(self.board), dtype=np.int)
        other_position[self.board == -1] = 1

        position = np.append(player1_position, other_position)

        id_ = ''.join(map(str, position))

        return id_

    @property
    def is_end_game(self) -> bool:
        if np.count_nonzero(self.board) == 42:
            return True

        for w in self.winners:
            if self.board[w].sum() == 4 * -self.player_turn:
                return True
        return False

    @property
    def value(self) -> Tuple[int, int, int]:
        # This is the value of the state for the current player
        # i.e. if the previous player played a winning move, you lose
        for w in self.winners:
            if self.board[w].sum() == 4 * -self.player_turn:
                return -1, -1, 1
        return 0, 0, 0

    def render(self, logger: Optional[Logger]) -> Optional[str]:
        if logger is not None:
            for r in range(6):
                logger.info([self.pieces[x] for x in self.board[7 * r: (7 * r + 7)]])
            logger.info('--------------')
            return None
        s = []
        for r in range(6):
            s.append(f"{[self.pieces[x] for x in self.board[7 * r: (7 * r + 7)]]}\n")
        s.append('--------------\n')
        return "".join(s)


# noinspection DuplicatedCode
class Game(AbstractGame[GameState]):
    grid_shape = np.array([6, 7])
    input_shape = np.array([2, 6, 7])
    action_size = 42
    name = 'connect4'
    identity_perm = np.array(
        [6, 5, 4, 3, 2, 1, 0, 13, 12, 11, 10, 9, 8, 7, 20, 19, 18, 17, 16, 15, 14, 27, 26, 25, 24, 23, 22, 21, 34, 33,
         32, 31, 30, 29, 28, 41, 40, 39, 38, 37, 36, 35])

    def __init__(self) -> None:
        super().__init__(current_player=1, game_state=GameState(np.zeros(42, dtype=np.int32), 1))
        self.state_size = len(self.game_state.binary)

    def reset(self) -> GameState:
        self.game_state = GameState(np.zeros(42, dtype=np.int32), 1)
        self.current_player = 1
        return self.game_state

    def step(self, action: int64) -> Tuple[AbstractGameState, float, int, None]:
        next_state, value, done = self.game_state.take_action(action)
        self.game_state = next_state
        self.current_player = -self.current_player
        info = None
        return next_state, value, done, info

    @staticmethod
    def identities(state: GameState, action_values: ndarray) -> List[Tuple[GameState, ndarray]]:
        identities = [(state, action_values)]

        current_board = state.board[Game.identity_perm]
        current_av = action_values[Game.identity_perm]

        identities.append((GameState(current_board, state.player_turn), current_av))

        return identities
