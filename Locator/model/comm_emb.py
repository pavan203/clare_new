from .gnn import GNNEncoder
import torch
import torch.nn as nn
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
        """with open('C:/Users/pavan/OneDrive/Desktop/function_calls.txt', 'a') as f:
            f.write(f"Function '{func.__name__}' from class '{class_name}' called {count} times\n")"""

        result = func(*args, **kwargs)
        return result

    return wrapper

class CommunityOrderEmbedding(nn.Module):
    @log_execution
    def __init__(self, args):
        super(CommunityOrderEmbedding, self).__init__()

        self.encoder = GNNEncoder(args)
        self.margin = args.margin
        self.device = args.device
    @log_execution
    def forward(self, emb_as, emb_bs):
        return emb_as, emb_bs
    @log_execution
    def predict(self, pred):
        emb_as, emb_bs = pred

        e = torch.sum(torch.max(torch.zeros_like(emb_as, device=self.device), emb_bs - emb_as) ** 2, dim=1)
        return e
    @log_execution
    def criterion(self, pred, labels):
        emb_as, emb_bs = pred
        e = torch.sum(torch.max(torch.zeros_like(emb_as, device=self.device), emb_bs - emb_as) ** 2, dim=1)
        margin = self.margin
        e[labels == 0] = torch.max(torch.tensor(0.0, device=self.device), margin - e)[labels == 0]
        return torch.sum(e)
