import networkx as nx
import numpy as np
from tqdm import tqdm
import torch
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
@log_execution
def feature_augmentation(nodes, edges, normalize=True):
    """Node feature augmentation `[deg(u), min(deg(N)), max(deg(N)), mean(deg(N)), std(deg(N))]` """
    g = nx.Graph(edges)
    g.add_nodes_from(nodes)

    num_node = len(nodes)

    node_degree = [g.degree[node] for node in range(num_node)]

    feat_matrix = np.zeros([num_node, 5], dtype=np.float32)
    feat_matrix[:, 0] = np.array(node_degree).squeeze()

    new_graph = nx.Graph()
    for node in tqdm(range(num_node), desc="Feature Computation"):
        if len(list(g.neighbors(node))) > 0:
            neighbor_deg = feat_matrix[list(g.neighbors(node)), 0]
            feat_matrix[node, 1:] = neighbor_deg.min(), neighbor_deg.max(), neighbor_deg.mean(), neighbor_deg.std()

    if normalize:
        feat_matrix = (feat_matrix - feat_matrix.mean(0, keepdims=True)) / (feat_matrix.std(0, keepdims=True) + 1e-9)

    for node in tqdm(range(num_node), desc="Feature Augmentation"):
        node_feat = feat_matrix[node, :].astype(np.float32)
        new_graph.add_node(node, node_feature=torch.from_numpy(node_feat))
    new_graph.add_edges_from(edges)
    return new_graph, feat_matrix


if __name__ == "__main__":
    ns = [3, 4, 5, 0, 2, 1]
    es = [[0, 1], [1, 2], [3, 4], [0, 2], [1, 3]]
    g, feats = feature_augmentation(ns, es, normalize=False)
    print(feats)
    print(g.nodes.data())
