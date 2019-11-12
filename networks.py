import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, num_actions, input_size, hidden_size, dueling = False):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.dueling = dueling
        if dueling:
            self.fc_value = nn.Linear(hidden_size, 1)
            self.fc_actions = nn.Linear(hidden_size, num_actions)
        else:
            self.fc3 = nn.Linear(hidden_size, self.num_actions)
    
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.view(x.size(0),-1)
        if not self.dueling:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            v = self.fc_value(x)
            a = self.fc_actions(x)
            x = a.add(v - a.mean(dim=-1).unsqueeze(-1))
        return x


class DQN_Conv(nn.Module):

    def __init__(self, h, w, output_size):
        super(DQN_Conv, self).__init__()
        self.conv1 = nn.Conv2d(4,16,kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        def conv2d_size_out(size, kernal_size=5, stride=2):
            return (size - kernal_size) // stride + 1
        func = conv2d_size_out
        convw = func(func(func(w)))
        convh = func(func(func(h)))
        fc_input = convh * convw * 32
        self.fc = nn.Linear(fc_input, output_size)
    
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.fc(x.view(x.size(0),-1))
        return x
