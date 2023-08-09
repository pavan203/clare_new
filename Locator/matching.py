import numpy as np
from tqdm import tqdm

from .model import CommunityOrderEmbedding
import random

from utils import sample_neigh, batch2graphs, generate_embedding, generate_ego_net, generate_embeddings

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

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


class CommMatching:
    @log_execution
    def __init__(self, args, graph, train_comms, val_comms):
        self.args = args

        self.graph = graph
        self.seen_nodes = {node for com in train_comms + val_comms for node in com}
        #print(train_comms,'\n',val_comms)
        self.train_comms, self.val_comms = self.init_comms(train_comms), self.init_comms(val_comms)
        #print('after\n',train_comms,'\n',val_comms)
        self.model = self.load_model()
        self.opt = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        #self.writer = SummaryWriter(args.writer_dir)
    @log_execution
    def load_model(self, load=False):
        """Load CommunityOrderEmbedding"""
        model = CommunityOrderEmbedding(self.args)
        model.to(self.args.device)
        if self.args.commm_path and load:
            model.load_state_dict(torch.load(self.args.commm_path, map_location=self.args.device))
        return model
    @log_execution
    def init_comms(self, comms):
        if len(comms) > 0:
            return [self.graph.subgraph(com) for com in comms if len(list(self.graph.subgraph(com).edges())) > 0]
        return []
    @log_execution
    def generate_batch(self, batch_size, valid=False, min_size=3, max_size=12):
        graphs = self.train_comms if not valid or len(self.val_comms) == 0 else self.val_comms



        pos_a, pos_b = [], []

        ratio = self.args.fine_ratio

        # Generate positive pairs
        for i in range(batch_size // 2):
            prob = random.random()
            if prob <= ratio:
                
                # Fine-grained sampling
                size = random.randint(min_size + 1, max_size)
                graph, a = sample_neigh(graphs, size)
                if len(a) - 1 <= min_size:
                    b = a
                else:
                    b = a[:random.randint(max(len(a) - 2, min_size), len(a))]
            else:
                #print('else',len(graphs))
                graph = None
                
                while graph is None or len(graph) < min_size + 1:    
                    graph = random.choice(graphs)
                
                a = graph.nodes

                _, b = sample_neigh([graph], random.randint(max(len(graph) - 2, min_size), len(graph)))
                # print(f"[Pos pair] Choose graph {a}, subgraph {b}")
            neigh_a, neigh_b = graph.subgraph(a), graph.subgraph(b)

            if len(neigh_a.edges()) > 0 and len(neigh_b.edges()) > 0:
                pos_a.append(neigh_a)
                pos_b.append(neigh_b)

        # Generate negative pairs
        neg_a, neg_b = [], []
        for i in range(batch_size // 2):
            prob = random.random()
            if prob <= ratio:
                size = random.randint(min_size + 1, max_size)
                graph_a, a = sample_neigh(graphs, random.randint(min_size, size))
                graph_b, b = sample_neigh(graphs, size)
            else:
                graph_b = None
                while graph_b is None or len(graph_b) < min_size + 1:
                    graph_b = random.choice(graphs)
                b = graph_b.nodes

                graph_a, a = sample_neigh(graphs, random.randint(min_size, len(graph_b)))
                # print(f"[Neg pair] Choose graph a{a}, graph b{b}")
            neigh_a, neigh_b = graph_a.subgraph(a), graph_b.subgraph(b)
            if len(neigh_a.edges()) > 0 and len(neigh_b.edges()) > 0:
                neg_a.append(neigh_a)
                neg_b.append(neigh_b)
        #print(pos_a[0].edges(),'\n',neg_a[0].edges())
        pos_a = batch2graphs(pos_a, device=self.args.device)
        pos_b = batch2graphs(pos_b, device=self.args.device)
        neg_a = batch2graphs(neg_a, device=self.args.device)
        neg_b = batch2graphs(neg_b, device=self.args.device)
        return pos_a, pos_b, neg_a, neg_b
    @log_execution
    def train_epoch(self, epochs):
        self.model.share_memory()

        batch_size = self.args.batch_size
        pairs_size = self.args.pairs_size
        device = self.args.device

        valid_set = []
        for _ in range(batch_size):
            pos_a, pos_b, neg_a, neg_b = self.generate_batch(pairs_size, valid=True)

            valid_set.append((pos_a, pos_b, neg_a, neg_b))

        """with open('C:/Users/pavan/OneDrive/Desktop/function_calls.txt', 'a') as f:
                f.write('\n\n normal generate over\n\n')"""

        for epoch in range(epochs):
            for batch in range(batch_size):
                self.model.train()

                pos_a, pos_b, neg_a, neg_b = self.generate_batch(pairs_size)
                emb_pos_a, emb_pos_b = self.model.encoder(pos_a), self.model.encoder(pos_b)
                emb_neg_a, emb_neg_b = self.model.encoder(neg_a), self.model.encoder(neg_b)

                emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
                emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)

                labels = torch.tensor([1] * pos_a.num_graphs + [0] * neg_a.num_graphs).to(device)
                pred = self.model(emb_as, emb_bs)

                self.model.zero_grad()
                loss = self.model.criterion(pred, labels)
                loss.backward()
                # torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.0)
                self.opt.step()

                if (batch + 1) % 5 == 0:
                    #self.writer.add_scalar(f"CommM Loss/Train", loss.item(), batch + epoch * batch_size)
                    print(f"Epoch {epoch + 1}, Batch{batch + 1}, Loss {loss.item():.4f}")
                if (batch + 1) % 10 == 0:
                    self.valid_model(valid_set, batch + epoch * batch_size)
        #torch.save(self.model.state_dict(), self.args.writer_dir + "/commm.pt")
    @log_execution
    def valid_model(self, valid_set, batch_num):
        """Test model on `valid_set`"""
        self.model.eval()
        device = self.args.device

        total_loss = 0
        for pos_a, pos_b, neg_a, neg_b in valid_set:
            labels = torch.tensor([1] * pos_a.num_graphs + [0] * neg_a.num_graphs).to(device)

            with torch.no_grad():
                emb_pos_a, emb_pos_b = self.model.encoder(pos_a), self.model.encoder(pos_b)
                emb_neg_a, emb_neg_b = self.model.encoder(neg_a), self.model.encoder(neg_b)

                emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
                emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)

                pred = self.model(emb_as, emb_bs)
                loss = self.model.criterion(pred, labels)
                total_loss += loss.item()
        total_loss /= len(valid_set)
        #self.writer.add_scalar(f"CommM Loss/Val", loss.item(), batch_num)
        print("[Eval-Test] Validation Loss{:.4f}".format(total_loss))

        # TODO: Save model
        # torch.save(self.model.state_dict(), self.args.writer_dir + "/commm.pt")
    @log_execution
    def load_embedding(self):
        query_emb = generate_embedding(self.train_comms + self.val_comms, self.model, device=self.args.device)
        print(len(self.graph.nodes()))
        n_node = len(list(self.graph.nodes()))
        batch_size = 10000 
        batch_len = int((n_node / batch_size) + 1)
        print('batch len',batch_len,n_node)

        all_emb = np.zeros((n_node, self.args.output_dim))
        for batch_num in range(batch_len):
            start, end = batch_num * batch_size, min((batch_num+1)*batch_size, n_node)
            print(start,end,'start end')
            gr=[]
            for i in range(start,end-1000,1000):

                graph = [generate_ego_net(self.graph, g, k=self.args.n_layers, max_size=self.args.comm_max_size) for g in
                      range(i, i+1000)]
                gr.append(graph)
            #graph2 = [generate_ego_net(self.graph, g, k=self.args.n_layers, max_size=self.args.comm_max_size) for g in range(end//2, end)]
            print('\ngenerate_ego_net done')
            #print('len of graphs',len(graph1),len(graph2))
            tmp_emb = generate_embeddings(gr, self.model, device=self.args.device)
            # all_emb = tmp_emb if batch_num == 0 else np.vstack((all_emb, tmp_emb))
            all_emb[start:end, :] = tmp_emb
            print(
                "No.{}-{} candidate com embedding finish".format(start, end))
        #np.save(self.args.writer_dir + "/emb", all_emb)
        #np.save(self.args.writer_dir + "/query", query_emb)
        return all_emb, query_emb
    @log_execution
    def make_prediction(self):
        all_emb, query_emb = self.load_embedding()
        print('len of embedings ',len(all_emb),len(query_emb))
        print(f"[Load Embedding], All shape {all_emb.shape}, Query shape {query_emb.shape}")

        pred_comms = []

        pred_size = self.args.pred_size
        single_pred_size = int(pred_size / query_emb.shape[0])
        print('single pred',single_pred_size,pred_size,query_emb.shape[0])

        seeds = []
        

        for j,i in enumerate(tqdm(range(query_emb.shape[0]), desc="Matching Communities")):
            q_emb = query_emb[i, :]
            distance = np.sqrt(np.sum(np.asarray(q_emb - all_emb) ** 2, axis=1))
            
            sort_dic = list(np.argsort(distance))
            print('dist sort',sort_dic[0:5])

            if len(pred_comms) >= pred_size:
                break

            length = 0
            for node in sort_dic:
                if length >= single_pred_size:
                    break
                neighs = generate_ego_net(self.graph, node, k=self.args.n_layers, max_size=self.args.comm_max_size,
                                          choice="neighbors")

                if neighs not in pred_comms and len(pred_comms) < pred_size and node not in self.seen_nodes and node \
                        not in seeds:
                    seeds.append(node)
                    pred_comms.append(neighs)
                    length += 1
                    #print(f"[Generate] seed node {node}, community {neighs}")

                                
        
        lengths = np.array([len(pred_com) for pred_com in pred_comms])
        #print(f"[Generate] Pred size {len(pred_comms)}, Avg Length {np.mean(lengths):.04f}")
        return pred_comms, all_emb