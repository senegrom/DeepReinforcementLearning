from abc import ABC, abstractmethod
from logging import Logger
from typing import Optional, Generic, TypeVar, Tuple

import numpy as np
import tensorflow as tf


class AbstractGameState(ABC):

    def __init__(self, board: np.ndarray, player_turn: int):
        self.player_turn: int = player_turn
        self.board: np.ndarray = board

    @property
    @abstractmethod
    def binary(self) -> tf.Tensor:
        pass

    @property
    @abstractmethod
    def is_end_game(self) -> bool:
        pass

    @property
    @abstractmethod
    def allowed_actions(self) -> list:
        pass

    @property
    @abstractmethod
    def value(self) -> Tuple[int, int, int]:
        pass

    @property
    @abstractmethod
    def id(self) -> str:
        pass

    @property
    def score(self) -> Tuple[int, int]:
        a, b, _ = self.value
        return a, b

    @abstractmethod
    def render(self, logger: Optional[Logger]) -> Optional[str]:
        pass

    def take_action(self, action: int) -> Tuple["AbstractGameState", float, int]:
        new_board = np.copy(self.board)
        new_board[action] = self.player_turn
        new_state = self.__class__(new_board, -self.player_turn)

        value = 0
        done = 0

        if new_state.is_end_game:
            value = new_state.value[0]
            done = 1

        return new_state, value, done


_TGameState = TypeVar('_TGameState', covariant=True, bound=AbstractGameState)


class AbstractGame(Generic[_TGameState], ABC):

    def __init__(self, current_player: int, game_state: _TGameState, action_size: int, name: str):
        self.current_player = current_player
        self.game_state: _TGameState = game_state
        self.name = name
        self.action_size = action_size
