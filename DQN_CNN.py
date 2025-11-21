import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Hyperparams
gamma = 0.99


class DQN(nn.Module):
    """Convolutional DQN that accepts a board plane (8x8) and an action plane (8x8).

    Minimal changes from the previous API: keep the same method names.
    `__call__(states, actions)` accepts a variety of small shapes:
      - states can be (B,64), (B,8,8), or (B,1,8,8) or a single (8,8)
      - actions can be (B,64), (B,8,8), or (B,1,8,8) or a single (8,8)

    The network concatenates state and action as two channels and applies a small CNN
    backbone followed by a tiny MLP head that outputs a single scalar Q(s,a).
    """
    def __init__(self, device=torch.device('cpu')) -> None:
        super().__init__()
        self.device = device

        # CNN backbone: input channels = 2 (state plane, action plane)
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=8)

        # LeakyReLU activation to avoid dead neurons
        self.leakyRelu = nn.LeakyReLU(negative_slope=0.01)

        # MLP head
        self.fc1 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

        self.MSELoss = nn.MSELoss()

    def forward(self, x):
        # x expected shape: (B, 2, 8, 8)
        x = self.leakyRelu(self.conv1(x))
        x = self.leakyRelu(self.conv2(x))
        x = self.leakyRelu(self.conv3(x))
        x = self.pool(x)  # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 128)
        x = self.leakyRelu(self.fc1(x))
        x = self.output(x)  # (B, 1)
        return x

    def loss(self, Q_value, rewards, Q_next_Values, Dones):
        Q_new = rewards + gamma * Q_next_Values * (1 - Dones)
        return self.MSELoss(Q_value, Q_new)

    def load_params(self, path):
        self.load_state_dict(torch.load(path))

    def save_params(self, path):
        torch.save(self.state_dict(), path)

    def copy(self):
        return copy.deepcopy(self)

    def __call__(self, states, actions):
        """Simplified: expect `states` and `actions` with shape (B, 8, 8).

        Both tensors are moved to `self.device`, converted to float, then
        expanded with a channel dimension and concatenated into (B,2,8,8)
        which is forwarded through the network.
        """
        states = states.to(self.device).float()
        actions = actions.to(self.device).float()

        s_plane = states.unsqueeze(1)  # (B,1,8,8)
        a_plane = actions.unsqueeze(1)  # (B,1,8,8)

        x = torch.cat((s_plane, a_plane), dim=1)  # (B,2,8,8)
        return self.forward(x)