import numpy as np
import torch
from Action import Action

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
        return self.board.sum()

    def __eq__(self, other) ->bool:
        return np.equal(self.board, other.board).all() 

    def __hash__(self) -> int:
        return hash(repr(self.board))
    
    def copy (self):
        newBoard = np.copy(self.board)
        legal_actions = self.legal_actions.copy()
        return State(board=newBoard, player=self.player, legal_actions=legal_actions)
    
    def reverse (self):
        reversed = self.copy()
        reversed.board = reversed.board * -1
        reversed.player = reversed.player * -1
        return reversed

    def toTensor (self, device = torch.device('cpu')) -> tuple:
        board_tensor = torch.from_numpy(self.board.astype(np.float32)).to(device)
        board_tensor = board_tensor.view(8, 8).unsqueeze(0)

        # Convert legal actions (list of (r,c)) into action planes (N,8,8)
        if len(self.legal_actions):
            actions_tensor = Action.coords_to_planes(self.legal_actions, device=device)
        else:
            actions_tensor = torch.empty((0, 8, 8), dtype=torch.float32, device=device)

        return board_tensor, actions_tensor
    
    @staticmethod
    def tensorToState (state_tuple, player):
        board_tensor = state_tuple[0]
        # expect (1,8,8)
        board = board_tensor.squeeze(0).cpu().numpy()

        legal_actions_tensor = state_tuple[1]
        legal_actions = []
        
        if legal_actions_tensor.numel() > 0:
            legal_actions = Action.planes_to_coords(legal_actions_tensor)

        return State(board, player=player, legal_actions=legal_actions)