import json
from Locator import CommMatching
from Rewriter import CommRewriting
import argparse
from datetime import datetime
import random
import numpy as np
import torch
from utils import load, feature_augmentation, split_communities, eval_scores
import os
import networkx as nx
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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
def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@log_execution
def write2file(comms, filename):
    with open(filename, 'w') as fh:
        content = '\n'.join([', '.join([str(i) for i in com]) for com in comms])
        fh.write(content)

@log_execution
def read4file(filename):
    with open(filename, "r") as file:
        pred = [[int(node) for node in x.split(', ')] for x in file.read().strip().split('\n')]
    return pred
import networkx as nx
import matplotlib.pyplot as plt
@log_execution
def visualize_graph(graph, pred_comms, communities):
    # Create an empty graph
    G = nx.Graph()

    # Add nodes to the graph
    for node in graph.nodes():
        G.add_node(node)

    # Add edges to the graph
    for edge in graph.edges():
        G.add_edge(edge[0], edge[1])

    # Create a dictionary to map nodes to their respective predicted community labels
    node_to_pred_comm = {}
    for i, comm in enumerate(pred_comms):
        for node in comm:
            node_to_pred_comm[node] = i

    # Create a dictionary to map nodes to their respective ground truth community labels
    node_to_true_comm = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_to_true_comm[node] = i

    # Get the number of predicted communities
    num_pred_comm = len(pred_comms)

    # Generate a color map for the predicted communities
    pred_comm_colors = [f"C{i}" for i in range(num_pred_comm)]

    # Generate a color map for the ground truth communities
    true_comm_colors = [f"C{i}" for i in range(len(communities))]

    # Draw the graph with predicted communities
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    node_colors = []
    for node in G.nodes():
        if node in node_to_pred_comm:
            node_colors.append(pred_comm_colors[node_to_pred_comm[node]])
        else:
            node_colors.append("gray")  # Handle missing nodes
    nx.draw_networkx(G, pos, node_color=node_colors, node_size=50, with_labels=False, cmap='tab20')
    plt.title("Graph with Predicted Communities")
    plt.show()

    # Draw the graph with ground truth communities
    plt.figure(figsize=(12, 8))
    node_colors = []
    for node in G.nodes():
        if node in node_to_true_comm:
            node_colors.append(true_comm_colors[node_to_true_comm[node]])
        else:
            node_colors.append("gray")  # Handle missing nodes
    nx.draw_networkx(G, pos, node_color=node_colors, node_size=50, with_labels=False, cmap='tab20')
    plt.title("Graph with Ground Truth Communities")
    plt.show()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
   

    # Community Locator related
    parser.add_argument("--conv_type", type=str, help="type of convolution", default="GCN")
    parser.add_argument("--n_layers", type=int, help="number of gnn layers", default=2)
    parser.add_argument("--hidden_dim", type=int, help="training hidden size", default=64)
    parser.add_argument("--output_dim", type=int, help="training hidden size", default=64)
    parser.add_argument("--dropout", type=float, help="dropout rate", default=0.2)
    parser.add_argument("--margin", type=float, help="margin loss", default=0.4)
    parser.add_argument("--fine_ratio", dest="fine_ratio", type=float, help="fine-grained sampling ratio", default=0.0)
    parser.add_argument("--comm_max_size", type=int, help="Community max size", default=4500)

    # Train Community Locator
    parser.add_argument("--lr", dest="lr", type=float, help="learning rate", default=1e-4)
    parser.add_argument("--device", dest="device", type=str, help="training device", default="cpu")
    parser.add_argument("--batch_size", type=int, help="training batch size", default=32)
    parser.add_argument("--pairs_size", type=int, help="pairs size", default=100)
    parser.add_argument("--seed", type=int, help="seed", default=0)

    parser.add_argument("--pred_size", type=int, help="pred size", default=45)
    parser.add_argument("--commm_path", type=str, help="CommM path", default="")
    parser.add_argument("--commr_path", type=str, help="CommR path", default="")

    parser.add_argument("--dataset", type=str, help="dataset", default="amazon")

    # Train Community Rewriter
    parser.add_argument("--agent_lr", type=float, help="CommR learning rate", default=1e-3)
    parser.add_argument("--n_episode", type=int, help="number of episode", default=10)
    parser.add_argument("--n_epoch", type=int, help="number of epoch", default=1000)
    parser.add_argument("--gamma", type=float, help="CommR gamma", default=0.99)
    parser.add_argument("--max_step", type=int, help="", default=10)


    #directory_path = 'C:/Users/pavan/OneDrive/Desktop/'
    # Save log
    parser.add_argument("--writer_dir", type=str, help="Summary writer directory", default="")

    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_all(args.seed)

    if not os.path.exists(f"{args.dataset}"):
        os.mkdir(f"{args.dataset}")
    args.writer_dir = f"{args.dataset}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    
    #args.comm_max_size = 20 if args.dataset.startswith("lj") else 12
    
    print(args.writer_dir)

    print('= ' * 20)
    print('##  Starting Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)

    nodes, edges, communities,r_m = load(args.dataset)
    comm=p_c=[[r_m[i] for i in j] for j in communities]
    g=[sorted(i) for i in comm]
    gg=sorted(g,key=lambda x:x[0])
    
    #c=[i.sorted()for i in comm]


    #print(sorted(nodes),'\n',edges,'\n',communities)

    """with open(directory_path + 'c.txt', 'w') as file1:
        for item in gg:
            file1.write(str(item) + '\n')"""
    graph, _ = feature_augmentation(nodes, edges)
    g=[sorted(i) for i in communities]
    gg=sorted(g,key=lambda x:x[0])
    train_comms, val_comms, test_comms = split_communities(gg, 8, 2)
    print('lengths',len(train_comms),len(val_comms),len(test_comms))
    

    # Training CommMatching of Locator
    CommM_obj = CommMatching(args, graph, train_comms, val_comms)
    """with open('C:/Users/pavan/OneDrive/Desktop/function_calls.txt', 'a') as f:
            f.write('\n\nCommMatching over\n\n')"""
    CommM_obj.train_epoch(1)
    """with open('C:/Users/pavan/OneDrive/Desktop/function_calls.txt', 'a') as f:
            f.write('\n\ntrain epoch over\n\n')"""
    
    pred_comms, feat_mat = CommM_obj.make_prediction()
    """with open('C:/Users/pavan/OneDrive/Desktop/function_calls.txt', 'a') as f:
            f.write('\n\npred over\n\n')"""
    p_c=[[r_m[i] for i in j] for j in pred_comms]

    # Save list1 to a text document
    g=[sorted(i) for i in p_c]
    gg=sorted(g,key=lambda x:x[0])
    """with open(directory_path + 'p.txt', 'w') as file1:
        for item in gg:
            file1.write(str(item) + '\n')"""

    # Save list2 to a text document

    """with open(directory_path + 't.txt', 'w') as file2:
        for item in test_comms:
            file2.write(str(item) + '\n')"""


    f, j, nmi = eval_scores(pred_comms, test_comms, tmp_print=True)
    metrics_string = '_'.join([f'{x:0.4f}' for x in [f, j, nmi]])
    write2file(pred_comms, args.writer_dir + "/CommM_.txt")
    
    #visualize_graph(graph, pred_comms, communities)

    # Use F1-score as Reward function
    # Train Community Rewriter
    cost_choice = "f1"
    # Note that feed predicted communities `pred_comms` into Community Rewriter
    CommR_obj = CommRewriting(args, graph, feat_mat, train_comms, val_comms, pred_comms, cost_choice)
    CommR_obj.train()
    rewrite_comms = CommR_obj.get_rewrite()
    r_c=[[r_m[i] for i in j] for j in rewrite_comms]
    g=[sorted(i) for i in p_c]
    gg=sorted(g,key=lambda x:x[0])
    """with open(directory_path + 'r_C.txt', 'w') as file1:
        for item in gg:
            file1.write(str(item) + '\n')"""
    f, j, nmi = eval_scores(rewrite_comms, test_comms, tmp_print=True)
    metrics_string = '_'.join([f'{x:0.4f}' for x in [f, j, nmi]])
    write2file(rewrite_comms, args.writer_dir + f"/CommR_{cost_choice}_" + metrics_string + '.txt')
    print('ans',len(rewrite_comms),len(pred_comms),len(communities))
    #visualize_graph(graph, rewrite_comms,pred_comms)
    # Save setting
    """with open(args.writer_dir + '/settings.json', 'w') as fh:
        arg_dict = vars(args)
        json.dump(arg_dict, fh, sort_keys=True, indent=4)"""

    print('## Finishing Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print('= ' * 20)
    """for i in range(1000):
        print(len(communities[i]),len(pred_comms[i]),len(rewrite_comms[i]))
"""


