from State import State
from Reversi import Reversi
import random

class Random_Agent:
    def __init__(self, env : Reversi, player = None) -> None:
        self.env = env

    def get_Action (self, events = None, graphics=None, state: State = None, epoch = 0, train = None):
            action = random.choice(state.legal_actions)
            return action
    