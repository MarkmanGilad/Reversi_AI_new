import numpy as np
import torch

class State:
    def __init__(self, board= None, player = 1, legal_actions = []) -> None:
        self.board = board
        self.player = player
        self.legal_actions = legal_actions 

    def get_opponent (self):
        return -self.player
                    
    def switch_player(self):
        self.player = self.get_opponent()

    def score (self):
        opponent = self.get_opponent()

        player_score = np.count_nonzero(self.board == self.player)
        opponent_score = np.count_nonzero(self.board == opponent)
        return player_score, opponent_score
        


    def __eq__(self, other) ->bool:
        return np.equal(self.board, other.board).all() 

    def __hash__(self) -> int:
        return hash(repr(self.board))
    
    def copy (self):
        newBoard = np.copy(self.board)
        legal_actions = self.legal_actions.copy()
        return State(board=newBoard, player=self.player, legal_actions=legal_actions)
    
    def toTensor (self, device = torch.device('cpu')) -> tuple:
        board_np = self.board.reshape(-1)
        board_tensor = torch.tensor(board_np, dtype=torch.float32, device=device)
        actions_np = np.array(self.legal_actions)
        actions_tensor = torch.from_numpy(actions_np)
        return board_tensor, actions_tensor
    
    [staticmethod]
    def tensorToState (state_tuple, player):
        board_tensor = state_tuple[0]
        board = board_tensor.reshape([8,8]).cpu().numpy()
        legal_actions_tensor = state_tuple[1]
        legal_actions = legal_actions_tensor.cpu().numpy()
        legal_actions = list(map(tuple, legal_actions))
        return State(board, player=player, legal_actions=legal_actions)