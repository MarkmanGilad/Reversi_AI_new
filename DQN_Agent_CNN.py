import torch
import random
import math
from DQN_CNN import DQN
from Constant import *
from State import State

class DQN_Agent:
    def __init__(self, player = 1, parametes_path = None, train = True, env= None):
        self.DQN = DQN()
        if parametes_path:
            self.DQN.load_params(parametes_path)
        self.player = player
        self.train = train
        self.setTrainMode()

    def setTrainMode (self):
          if self.train:
              self.DQN.train()
          else:
              self.DQN.eval()

    def get_Action (self, state:State, epoch = 0, events= None, train = True, graphics = None, black_state = None) -> tuple:
        actions = state.legal_actions
        if self.train and train:
            epsilon = self.epsilon_greedy(epoch)
            rnd = random.random()
            if rnd < epsilon:
                return random.choice(actions)
        
        # Get board tensor (1,8,8) and action planes (N,8,8)
        state_tensor, action_planes = state.toTensor()

        n_actions = int(action_planes.size(0))

        # Repeat state to shape (N,8,8) and move tensors to model device
        state_batch = state_tensor.repeat((n_actions, 1, 1)).to(self.DQN.device).float()
        action_batch = action_planes.to(self.DQN.device).float()

        with torch.no_grad():
            Q_values = self.DQN(state_batch, action_batch)  # (N,1)
        Q_values = Q_values.view(-1)
        max_index = int(torch.argmax(Q_values).item())
        return actions[max_index]

    def get_Actions (self, states_tensor: State, dones) -> torch.tensor:
        actions = []
        boards_tensor = states_tensor[0]
        actions_tensor = states_tensor[1]
        for i, board in enumerate(boards_tensor):
            if dones[i].item():
                actions.append((0,0))
            else:
                actions.append(self.get_Action(State.tensorToState(state_tuple=(boards_tensor[i],actions_tensor[i]),player=self.player), train=False))
        return torch.tensor(actions)

    def epsilon_greedy(self,epoch, start = epsilon_start, final=epsilon_final, decay=epsilon_decay):
        if epoch >= decay:
            return final
        return start + (final - start) * (epoch / float(decay))
    
    def loadModel (self, file):
        self.model = torch.load(file)
    
    def save_param (self, path):
        self.DQN.save_params(path)

    def load_params (self, path):
        self.DQN.load_params(path)

    def __call__(self, events= None, state=None):
        return self.get_Action(state)
