import numpy as np
from Reversi import Reversi
from State import State
import random
from Constant import OPPONENT_EPSILON_START, OPPONENT_EPSILON_FINAL, OPPONENT_EPSILON_DECAY


class Fix_Agent:
    def __init__(self, env, player = 1, train = False, random = OPPONENT_EPSILON_START) -> None:
        self.env  = env
        self.player = player
        self.train = train
        self.random = random

    def epsilon_decay(self, epoch):
        """Calculate the epsilon value for the opponent based on curriculum learning schedule."""
        if epoch >= OPPONENT_EPSILON_DECAY:
            return OPPONENT_EPSILON_FINAL
        else:
            return OPPONENT_EPSILON_START + (OPPONENT_EPSILON_FINAL - OPPONENT_EPSILON_START) * (epoch / OPPONENT_EPSILON_DECAY)

    def value(self, state: State):
        v = np.array([[100, -25, 10, 5, 5, 10, -25, 100], 
                    [-25, -25, 2, 2, 2, 2, -25, -25],
                    [10, 2, 5, 1, 1, 5, 2, 10],
                    [5,2,1,2,2,1,2,5],
                    [5,2,1,2,2,1,2,5],
                    [10, 2, 5, 1, 1, 5, 2, 10],
                    [-25, -25, 2, 2, 2, 2, -25, -25],
                    [100, -25, 10, 5, 5, 10, -25, 100]])
        board = state.board
        return (board*v).sum()
        
    def get_Action (self, events = None, graphics=None, state: State = None, epoch = 0, train = True):
        legal_actions = state.legal_actions
        if self.train and train:
            current_epsilon = self.epsilon_decay(epoch)
            if random.random() < current_epsilon:
                return random.choice(legal_actions)
        next_states, _ = self.env.get_all_next_states(state)
        values = []
        for next_state in next_states:
                values.append(self.value(next_state))
        if self.player == 1:
            maxIndex = values.index(max(values))
            return legal_actions[maxIndex]
        else:
            minIndex = values.index(min(values))
            return legal_actions[minIndex]

    
       
        