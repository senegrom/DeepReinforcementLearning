from typing import Optional

import numpy as np


class Game:

    def __init__(self):
        self.current_player = 1
        self.game_state = GameState(np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0], dtype=np.int), 1)
        self.action_space = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0], dtype=np.int)
        self.pieces = {'1': 'X', '0': '-', '-1': 'O'}
        self.grid_shape = (6, 7)
        self.input_shape = (2, 6, 7)
        self.name = 'connect4'
        self.state_size = len(self.game_state.binary)
        self.action_size = len(self.action_space)

    def reset(self):
        self.game_state = GameState(np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0], dtype=np.int), 1)
        self.current_player = 1
        return self.game_state

    def step(self, action):
        next_state, value, done = self.game_state.take_action(action)
        self.game_state = next_state
        self.current_player = -self.current_player
        info = None
        return next_state, value, done, info

    @staticmethod
    def identities(state: "GameState", action_values):
        identities = [(state, action_values)]

        current_board = state.board
        current_av = action_values

        current_board = np.array([
            current_board[6], current_board[5], current_board[4], current_board[3], current_board[2], current_board[1],
            current_board[0], current_board[13], current_board[12], current_board[11], current_board[10],
            current_board[9], current_board[8], current_board[7], current_board[20], current_board[19],
            current_board[18], current_board[17], current_board[16], current_board[15], current_board[14],
            current_board[27], current_board[26], current_board[25], current_board[24], current_board[23],
            current_board[22], current_board[21], current_board[34], current_board[33], current_board[32],
            current_board[31], current_board[30], current_board[29], current_board[28], current_board[41],
            current_board[40], current_board[39], current_board[38], current_board[37], current_board[36],
            current_board[35]
        ])

        current_av = np.array([
            current_av[6], current_av[5], current_av[4], current_av[3], current_av[2], current_av[1], current_av[0],
            current_av[13], current_av[12], current_av[11], current_av[10], current_av[9], current_av[8], current_av[7],
            current_av[20], current_av[19], current_av[18], current_av[17], current_av[16], current_av[15],
            current_av[14], current_av[27], current_av[26], current_av[25], current_av[24], current_av[23],
            current_av[22], current_av[21], current_av[34], current_av[33], current_av[32], current_av[31],
            current_av[30], current_av[29], current_av[28], current_av[41], current_av[40], current_av[39],
            current_av[38], current_av[37], current_av[36], current_av[35]
        ])

        identities.append((GameState(current_board, state.player_turn), current_av))

        return identities


class GameState:
    def __init__(self, board, player_turn):
        self.board = board
        self.pieces = {'1': 'X', '0': '-', '-1': 'O'}
        self.winners = [
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
        self.player_turn = player_turn
        self.binary = self._binary()
        self.id = self._convert_state_to_id()
        self.allowed_actions = self._allowed_actions()
        self.is_end_game = self._check_for_end_game()
        self.value = self._get_value()
        self.score = self._get_score()

    def _allowed_actions(self):
        allowed = []
        for i in range(len(self.board)):
            if i >= len(self.board) - 7:
                if self.board[i] == 0:
                    allowed.append(i)
            else:
                if self.board[i] == 0 and self.board[i + 7] != 0:
                    allowed.append(i)

        return allowed

    def _binary(self):

        currentplayer_position = np.zeros(len(self.board), dtype=np.int)
        currentplayer_position[self.board == self.player_turn] = 1

        other_position = np.zeros(len(self.board), dtype=np.int)
        other_position[self.board == -self.player_turn] = 1

        position = np.append(currentplayer_position, other_position)

        return position

    def _convert_state_to_id(self):
        player1_position = np.zeros(len(self.board), dtype=np.int)
        player1_position[self.board == 1] = 1

        other_position = np.zeros(len(self.board), dtype=np.int)
        other_position[self.board == -1] = 1

        position = np.append(player1_position, other_position)

        id_ = ''.join(map(str, position))

        return id_

    def _check_for_end_game(self):
        if np.count_nonzero(self.board) == 42:
            return 1

        for x, y, z, a in self.winners:
            if self.board[x] + self.board[y] + self.board[z] + self.board[a] == 4 * -self.player_turn:
                return 1
        return 0

    def _get_value(self):
        # This is the value of the state for the current player
        # i.e. if the previous player played a winning move, you lose
        for x, y, z, a in self.winners:
            if self.board[x] + self.board[y] + self.board[z] + self.board[a] == 4 * -self.player_turn:
                return -1, -1, 1
        return 0, 0, 0

    def _get_score(self):
        tmp = self.value
        return tmp[1], tmp[2]

    def take_action(self, action) -> ("GameState", float, int):
        new_board = np.array(self.board)
        new_board[action] = self.player_turn

        new_state = GameState(new_board, -self.player_turn)

        value = 0
        done = 0

        if new_state.is_end_game:
            value = new_state.value[0]
            done = 1

        return new_state, value, done

    def render(self, logger) -> Optional[str]:
        if logger is not None:
            for r in range(6):
                logger.info([self.pieces[str(x)] for x in self.board[7 * r: (7 * r + 7)]])
            logger.info('--------------')
            return None
        s = []
        for r in range(6):
            s.append(f"{[self.pieces[str(x)] for x in self.board[7 * r: (7 * r + 7)]]}\n")
        s.append('--------------\n')
        return "".join(s)
