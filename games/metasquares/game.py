from typing import Optional

import numpy as np

from abstractgame import AbstractGame, AbstractGameState


class GameState(AbstractGameState):
    winners = [
        {'points': 1, 'tiles': [
            [0, 1, 5, 6],
            [1, 2, 6, 7],
            [2, 3, 7, 8],
            [3, 4, 8, 9],
            [5, 6, 10, 11],
            [6, 7, 11, 12],
            [7, 8, 12, 13],
            [8, 9, 13, 14],
            [10, 11, 15, 16],
            [11, 12, 16, 17],
            [12, 13, 17, 18],
            [13, 14, 18, 19],
            [15, 16, 20, 21],
            [16, 17, 21, 22],
            [17, 18, 22, 23],
            [18, 19, 23, 24]
        ]},
        {'points': 2, 'tiles': [
            [1, 5, 7, 11],
            [2, 6, 8, 12],
            [3, 7, 9, 13],
            [6, 10, 12, 16],
            [7, 11, 13, 17],
            [8, 12, 14, 18],
            [11, 15, 17, 21],
            [12, 16, 18, 22],
            [13, 17, 19, 23]
        ]},
        {'points': 4, 'tiles': [
            [0, 2, 10, 12],
            [1, 3, 11, 13],
            [2, 4, 12, 14],
            [5, 7, 15, 17],
            [6, 8, 16, 18],
            [7, 9, 17, 19],
            [10, 12, 20, 22],
            [11, 13, 21, 23],
            [12, 14, 22, 24]
        ]},
        {'points': 5, 'tiles': [
            [1, 10, 8, 17],
            [6, 15, 13, 22],
            [2, 11, 9, 18],
            [7, 16, 14, 23],
            [2, 5, 13, 16],
            [7, 10, 18, 21],
            [3, 6, 14, 17],
            [8, 11, 19, 22]
        ]},
        {'points': 8, 'tiles': [
            [2, 10, 14, 22]
        ]},
        {'points': 9, 'tiles': [
            [0, 3, 15, 18],
            [1, 4, 16, 19],
            [5, 8, 20, 23],
            [6, 9, 21, 24]
        ]},
        {'points': 10, 'tiles': [
            [1, 9, 23, 15],
            [5, 3, 19, 21]
        ]},
        {'points': 16, 'tiles': [
            [0, 4, 20, 24]
        ]},
    ]
    pieces = {'1': 'X', '0': '-', '-1': 'O'}

    def __init__(self, board, player_turn):
        super().__init__(board, player_turn)

    @property
    def allowed_actions(self) -> list:
        return np.where(self.board == 0)[0]

    @property
    def binary(self):

        currentplayer_position = np.zeros(len(self.board), dtype=np.int)
        currentplayer_position[self.board == self.player_turn] = 1

        other_position = np.zeros(len(self.board), dtype=np.int)
        other_position[self.board == -self.player_turn] = 1

        position = np.append(currentplayer_position, other_position)

        return position

    @property
    def id(self):
        player1_position = np.zeros(len(self.board), dtype=np.int)
        player1_position[self.board == 1] = 1

        other_position = np.zeros(len(self.board), dtype=np.int)
        other_position[self.board == -1] = 1

        position = np.append(player1_position, other_position)

        id_ = ''.join(map(str, position))

        return id_

    @property
    def is_end_game(self) -> bool:
        if np.count_nonzero(self.board) == 24:
            return True
        return False

    @property
    def value(self) -> (int, int, int):
        current_player_points = 0
        for square_type in self.winners:
            points = square_type['points']
            for tiles in square_type['tiles']:
                check_flag = 0
                tilenum = 0
                while tilenum < 4 and check_flag == 0:
                    if self.board[tiles[tilenum]] != self.player_turn:
                        check_flag = 1
                    tilenum += 1
                if check_flag == 0:
                    current_player_points += points

        opponent_player_points = 0
        for square_type in self.winners:
            points = square_type['points']
            for tiles in square_type['tiles']:
                check_flag = 0
                tilenum = 0
                while tilenum < 4 and check_flag == 0:
                    if self.board[tiles[tilenum]] != -self.player_turn:
                        check_flag = 1
                    tilenum += 1
                if check_flag == 0:
                    opponent_player_points += points

        if current_player_points > opponent_player_points:
            return 1, current_player_points, opponent_player_points
        elif current_player_points < opponent_player_points:
            return -1, current_player_points, opponent_player_points
        else:
            return 0, current_player_points, opponent_player_points

    def render(self, logger) -> Optional[str]:
        if logger is not None:
            for r in range(5):
                logger.info([self.pieces[str(x)] for x in self.board[5 * r: (5 * r + 5)]])
            logger.info('--------------')
            return None
        s = []
        for r in range(5):
            s.append(f"{[self.pieces[str(x)] for x in self.board[5 * r: (5 * r + 5)]]}\n")
        s.append('--------------\n')
        return "".join(s)


class Game(AbstractGame[GameState]):
    grid_shape = (5, 5)
    input_shape = (2, 5, 5)

    def __init__(self):
        super().__init__(1, GameState(np.zeros(25, dtype=np.int), 1), 25, 'metaSquares')
        self.state_size = len(self.game_state.binary)

    def reset(self):
        self.game_state = GameState(np.zeros(25, dtype=np.int), 1)
        self.current_player = 1
        return self.game_state

    def step(self, action):
        next_state, value, done = self.game_state.take_action(action)
        self.game_state = next_state
        self.current_player = -self.current_player
        info = None
        return next_state, value, done, info

    @staticmethod
    def identities(state: GameState, action_values):
        identities = []
        current_board = state.board
        current_av = action_values

        for _ in range(5):
            current_board = np.array([
                current_board[20], current_board[15], current_board[10], current_board[5], current_board[0],
                current_board[21], current_board[16], current_board[11], current_board[6], current_board[1],
                current_board[22], current_board[17], current_board[12], current_board[7], current_board[2],
                current_board[23], current_board[18], current_board[13], current_board[8], current_board[3],
                current_board[24], current_board[19], current_board[14], current_board[9], current_board[4]
            ])

            current_av = np.array([
                current_av[20], current_av[15], current_av[10], current_av[5], current_av[0],
                current_av[21], current_av[16], current_av[11], current_av[6], current_av[1],
                current_av[22], current_av[17], current_av[12], current_av[7], current_av[2],
                current_av[23], current_av[18], current_av[13], current_av[8], current_av[3],
                current_av[24], current_av[19], current_av[14], current_av[9], current_av[4]

            ])

            identities.append((GameState(current_board, state.player_turn), current_av))

        current_board = np.array([
            current_board[4], current_board[3], current_board[2], current_board[1], current_board[0],
            current_board[9], current_board[8], current_board[7], current_board[6], current_board[5],
            current_board[14], current_board[13], current_board[12], current_board[11], current_board[10],
            current_board[19], current_board[18], current_board[17], current_board[16], current_board[15],
            current_board[24], current_board[23], current_board[22], current_board[21], current_board[20]
        ])

        current_av = np.array([
            current_av[4], current_av[3], current_av[2], current_av[1], current_av[0],
            current_av[9], current_av[8], current_av[7], current_av[6], current_av[5],
            current_av[14], current_av[13], current_av[12], current_av[11], current_av[10],
            current_av[19], current_av[18], current_av[17], current_av[16], current_av[15],
            current_av[24], current_av[23], current_av[22], current_av[21], current_av[20]

        ])

        for _ in range(5):
            current_board = np.array([
                current_board[20], current_board[15], current_board[10], current_board[5], current_board[0],
                current_board[21], current_board[16], current_board[11], current_board[6], current_board[1],
                current_board[22], current_board[17], current_board[12], current_board[7], current_board[2],
                current_board[23], current_board[18], current_board[13], current_board[8], current_board[3],
                current_board[24], current_board[19], current_board[14], current_board[9], current_board[4]
            ])

            current_av = np.array([
                current_av[20], current_av[15], current_av[10], current_av[5], current_av[0],
                current_av[21], current_av[16], current_av[11], current_av[6], current_av[1],
                current_av[22], current_av[17], current_av[12], current_av[7], current_av[2],
                current_av[23], current_av[18], current_av[13], current_av[8], current_av[3],
                current_av[24], current_av[19], current_av[14], current_av[9], current_av[4]

            ])

            identities.append((GameState(current_board, state.player_turn), current_av))

        return identities
