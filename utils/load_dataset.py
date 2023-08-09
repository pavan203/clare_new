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

@log_execution
def load(name):
    """Load snap dataset"""
    communities = open(f"./dataset/{name}/{name}-1.90.cmty.txt")
    edges = open(f"./dataset/{name}/{name}-1.90.ungraph.txt")

    communities = [[int(i) for i in x.split()] for x in communities]
    edges = [[int(i) for i in e.split()] for e in edges]
    edges = [[u, v] if u < v else [v, u] for u, v in edges if u != v]

    nodes = {node for e in edges for node in e}
    mapping = {u: i for i, u in enumerate(sorted(nodes))}

    edges = [[mapping[u], mapping[v]] for u, v in edges]
    communities = [[mapping[node] for node in com] for com in communities]

    print(f"[{name.upper()}], #Nodes {len(nodes)}, #Edges {len(edges)} #Communities {len(communities)}")
    r_m={k:v for v,k in mapping.items()}
    return nodes, edges, communities,r_m
