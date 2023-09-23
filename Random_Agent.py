from State import State
from Reversi import Reversi
import random

class RandomAgent:
    def __init__(self, env : Reversi, player = None) -> None:
        self.env = env

    def get_Action (self, events = None, graphics=None, state: State = None, epoch = 0, train = None):
            action = random.choice(state.legal_actions)
            return action

    def get_state_action(self, event = None, graphics=None, state: State = None, epoch = 0):
        legal_actions = self.env.get_legal_actions(state)
        index = random.randint(0, len(legal_actions)-1)
        next_state = self.env.get_next_state(action=legal_actions[index],state = state)
        return next_state.toTensor(),legal_actions[index]