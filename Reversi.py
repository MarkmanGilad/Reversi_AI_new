import numpy as np
import torch
from State import State
from Constant import *

class Reversi:
    def __init__(self, state:State = None) -> None:
        if state is None:
            self.state = self.get_init_state((ROWS, COLS))
        else:
            self.state = state

    def get_init_state(self, Rows_Cols = (ROWS, COLS)):
        rows, cols = Rows_Cols
        board = np.zeros([rows, cols],int)
        board[3][3] = 1
        board[3][4] = -1
        board[4][3] = -1
        board[4][4] = 1
        legal_actions = [(3,5), (5,3), (4,2), (2, 4)]
        return State (board, player=1, legal_actions=legal_actions)
    
    def is_inside(self, row_col, state: State):
        row, col = row_col
        board_row, board_col = state.board.shape
        return 0 <= row < board_row and 0 <= col < board_col
    
    def flip_piece(self, row_col, state: State):
        row, col = row_col
        state.board[row][col] *= -1

    def check_legal_line(self, start_row_col: tuple[int, int], dir_row_col: tuple[int, int], state: State):
        count = 0
        opponent = state.get_opponent()
        row, col = start_row_col
        dir_row, dir_col = dir_row_col
        go = True
        while (go):
            row += dir_row
            col += dir_col
            if self.is_inside((row, col), state) and state.board[row, col] == opponent:
                count +=1
            else:
                go = False

        if self.is_inside((row, col), state) and state.board[row, col] == state.player and count > 0:
            return count
        return -1

    def is_legal_move(self, row_col, state: State):
        row, col = row_col
        if state.board[row][col] !=0:
            return False
        directions = (-1 , 0 , 1)
        for dir_row in directions:
            for dir_col in directions:
                if dir_row == dir_col == 0:
                    continue
                count = self.check_legal_line((row, col), (dir_row, dir_col), state)
                if  count > 0:
                    return True
        return False

    def reverse_line (self, row_col, dir_row_col, count, state: State) -> None:
        row, col = row_col
        dir_row, dir_col = dir_row_col
        for i in range(count):
            row += dir_row
            col += dir_col
            self.flip_piece((row, col), state)

    def move(self, action: tuple[int, int], state: State) -> bool:
        row, col = action
        directions = (-1 , 0 , 1)
        if state.board[row][col] !=0:
            return False
        legal = False
        for dir_row in directions:
            for dir_col in directions:
                if dir_row == dir_col == 0:
                    continue
                count = self.check_legal_line((row, col), (dir_row, dir_col), state)
                if  count > 0:
                    legal = True
                    self.reverse_line((row, col), (dir_row, dir_col), count, state)
        if legal:
            state.board[row, col] = state.player
            state.switch_player()
            self.set_legal_actions(state)
        return legal
    
    def set_legal_actions(self, state: State):
        legal_actions = []
        rows, cols = state.board.shape
        for row in range(rows):
            for col in range(cols):
                if self.is_legal_move((row,col), state):
                    legal_actions.append((row, col))
        if len(legal_actions)==0: #redundent state.legak_actions = legal_actions
            state.legal_actions = []
        else:
            state.legal_actions = legal_actions

    def get_legal_actions(self, state:State) -> list:
        return state.legal_actions
    
    def is_end_of_game(self, state: State) -> bool:
        if state.legal_actions:
            return False
        return True
    
    def get_next_state(self, action, state:State) -> State:
        next_state = state.copy()
        self.move(action, next_state)
        self.set_legal_actions(next_state)
        return next_state

    def get_all_next_states (self, state: State) -> tuple:
        legal_actions = state.legal_actions
        next_states = []
        for action in legal_actions:
            next_states.append(self.get_next_state(action, state))
        return next_states, legal_actions

    def toTensor (self, list_states, device = torch.device('cpu')) -> tuple:
        list_board_tensors = []
        list_legal_actions = []
        for state in list_states:
            board_tensor, legal_actions = state.toTensor(device) 
            list_board_tensors.append(board_tensor)
            list_legal_actions.append(torch.tensor(legal_actions))
        return torch.vstack(list_board_tensors), torch.vstack(list_legal_actions)
    
    def reward (self, state : State, action = None) -> tuple:
        if action:
            next_state = self.get_next_state(action, state)
        else:
            next_state = state
        if (self.is_end_of_game(next_state)):
            sum =  next_state.board.sum()
            if sum > 0:
                return 1, True  
            elif sum < 0:
                return -1, True  
            else:
                return 0, True  
        return 0, False
    