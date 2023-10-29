import numpy as np
from Reversi import Reversi
from State import State
import random


class Fix_Agent:
    def __init__(self, env, player = 1, train = False, random = 0.10) -> None:
        self.env  = env
        self.player = player
        self.train = train
        self.random = random

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
        if self.train and train and random.random() < self.random:
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

    
       
        