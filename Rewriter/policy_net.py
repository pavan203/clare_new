import torch.nn as nn
import torch
from torch_geometric.nn import GINConv

import logging
import functools
from collections import defaultdict
import inspect
logging.basicConfig(level=logging.DEBUG)



def log_execution(func):
    call_counts = defaultdict(int)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get the class name
        class_name = ""
        if inspect.stack()[1][3] == "<module>":
            # Function is called from the module level
            class_name = "Module"
        else:
            # Function is called from a class
            class_name = args[0].__class__.__name__

        call_counts[(class_name, func.__name__)] += 1
        count = call_counts[(class_name, func.__name__)]
        
        # Store the information in a meaningful way to a text file
        with open('C:/Users/pavan/OneDrive/Desktop/function_calls.txt', 'a') as f:
            f.write(f"Function '{func.__name__}' from class '{class_name}' called {count} times\n")

        result = func(*args, **kwargs)
        return result

    return wrapper
# Embedding updater
class GCN(nn.Module):
    @log_execution
    def __init__(self, input_channels=65, hidden_channels=64):
        super(GCN, self).__init__()
        # self.conv = GCNConv(input_channels, hidden_channels)
        self.conv = GINConv(
            nn.Sequential(nn.Linear(input_channels, hidden_channels), nn.ReLU(),
                          nn.Linear(hidden_channels, hidden_channels))
        )

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x


class MLP(nn.Module):
    @log_execution
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    @log_execution
    def forward(self, x):
        hid = torch.tanh(self.fc1(x))
        hid = torch.tanh(self.fc2(hid))
        return self.fc3(hid)
