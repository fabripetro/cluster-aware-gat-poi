import torch
import matplotlib
import matplotlib.pyplot as plt
import statistics
import numpy as np
from torch import nn
from sklearn.cluster import KMeans
import matplotlib.ticker as mticker

# Install required packages.
import os
import torch
import chardet
import pandas as pd
from collections import defaultdict
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.nn import Linear, Parameter
import random
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, to_dense_adj
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
import seaborn as sns
# Helper function for visualization.
import networkx as nx


def upload_dataset(name: str):
    data_set = torch.load(name)
    return data_set


def train_val_test_split_tensor(
        dataset: torch.Tensor,
        train_perc: float = 0.60,
        validation_perc: float = 0.20,
        test_perc: float = 0.20
):
    length = dataset.size(0)
    indices = torch.randperm(length)

    train_cutoff = int(length * train_perc)
    val_cutoff = train_cutoff + int(length * validation_perc)

    train_dataset = dataset[indices[:train_cutoff]]
    validation_dataset = dataset[indices[train_cutoff:val_cutoff]]
    test_dataset = dataset[indices[val_cutoff:]]

    return train_dataset, validation_dataset, test_dataset


def choose_K(
        dataset: list,
        max_K: int = 26,
        name: str = 'image'
):
    inertia = []
    K = range(1, max_K)
    for k in K:
        kmeans = KMeans(n_clusters=k, n_init='auto', max_iter=500, random_state=0).fit(dataset)
        inertia.append(kmeans.inertia_)
    matplotlib.use('Agg')
    plt.figure(figsize=(8, 4))
    plt.plot(K, inertia, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia (WCSS)')
    plt.title('Elbow Method For Optimal k')
    plt.grid()
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(base=1))  # Force ticks every 1 unit
    # plt.show()
    plt.savefig(f'{name}.png')


def K_Means(
        dataset: list,
        n_clusters: int = 5,
        max_iter: int = 500,
        random_state: int = 0,
):
    k_means = KMeans(
        n_clusters=n_clusters,
        n_init='auto',
        max_iter=max_iter,
        random_state=random_state
    )
    labels = k_means.fit_predict(dataset)
    # Print the number of elements of each cluster
    unique_values, counts = np.unique(labels, return_counts=True)
    frequency_dict = dict(zip(unique_values, counts))
    print("Label's Elements : ", frequency_dict)
    ids = []
    for el in dataset:
        ids.append(el[0])
    return ids, labels


def printProgressAction(
        action: str,
        iteration: int
):
    print(f'\r{action} {iteration}', end=" ")


def EarlyStopping(
        curr_monitor: float,
        old_monitor: float,
        count: int,
        patience: int = 5,
        min_delta: float = 0.0
):
    stop = False
    if (curr_monitor - old_monitor) <= min_delta:
        count += 1
        if count > patience:
            stop = True
            return count, stop
    count = 0
    return count, stop


def recall_k(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        k: int = 1
):
    top_k_pred = torch.topk(y_pred, k=k, dim=2).indices  # [batch_size, num_classes, k]
    y_true_expanded = y_true.unsqueeze(2).expand_as(top_k_pred)
    hits = (top_k_pred == y_true_expanded).any(dim=2).float()  # [batch_size, num_classes]
    recall = hits.mean().item()
    return recall


def init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1.414)
            if m.bias is not None:
                nn.init.constant_(m.bias, 1)
        elif isinstance(m, nn.Parameter):
            nn.init.xavier_uniform_(m.data, gain=1.414)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data, gain=1.414)
                elif 'weight_hh' in name:
                    nn.init.xavier_uniform_(param.data, gain=1.414)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 1)


def visualize_graph(
        G,
        x,
        color: str,
        name: str = 'image'
):
    matplotlib.use('Agg')
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])

    node_size = 30

    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False, node_color=color, cmap="Set2", node_size=node_size)
    plt.savefig(f'{name}.png')


def traj_graph(
        embedded_check_in_tuples_history: torch.Tensor,
        k: int = 1,
        cluster_discriminator: int = 0,
        device=torch.device("cpu")
):
    G = nx.MultiDiGraph()
    node_mapping = {}
    node_counter = 0

    for u_idx in range(embedded_check_in_tuples_history.size(0)):
        prev_POI = None
        for check_in_idx in range(embedded_check_in_tuples_history.size(1)):
            cat, pos, lab = embedded_check_in_tuples_history[u_idx, check_in_idx]

            node_key = (cat, pos)
            curr_POI = 0
            if len(node_mapping.keys()) == 0:
                node_mapping[node_key] = node_counter
                node_counter += 1
                cat_expanded = cat.unsqueeze(0)
                pos_expanded = pos.unsqueeze(0)
                G.add_node(node_mapping[node_key], embedding=torch.cat([cat_expanded, pos_expanded], dim=-1), label=lab)
                curr_POI = node_mapping[node_key]
            else:
                add_node_flag = False
                for key in node_mapping.keys():
                    if not(torch.equal(node_key[0], key[0]) and torch.equal(node_key[1], key[1])):
                        add_node_flag = True
                    else:
                        add_node_flag = False
                        curr_POI = node_mapping[key]
                        break
                if add_node_flag:
                    node_mapping[node_key] = node_counter
                    node_counter += 1
                    cat_expanded = cat.unsqueeze(0)
                    pos_expanded = pos.unsqueeze(0)
                    G.add_node(
                        node_mapping[node_key],
                        embedding=torch.cat([cat_expanded, pos_expanded], dim=-1),
                        label=lab
                    )
                    curr_POI = node_mapping[node_key]

            if prev_POI is not None:
                G.add_edge(prev_POI, curr_POI)
            prev_POI = curr_POI

    nodes = list(G.nodes())
    n = len(nodes)
    G_new = nx.MultiDiGraph()
    G_new.add_nodes_from(G.nodes(data=True))
    for i_index in range(n):
        for j_index in range(n):
            if i_index == j_index:
                continue
            u = nodes[i_index]
            v = nodes[j_index]
            existing_connections = G.number_of_edges(u, v)
            if existing_connections >= k:
                # Local Pattern Learning Module
                if cluster_discriminator == 0 and torch.equal(G_new.nodes[i_index]['label'],
                                                              G_new.nodes[j_index]['label']):
                    G_new.add_edge(u, v)
                # General Pattern Learning Module
                elif cluster_discriminator == 1 and not torch.equal(G_new.nodes[i_index]['label'],
                                                                    G_new.nodes[j_index]['label']):
                    G_new.add_edge(u, v)

    edges = list(G_new.edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.stack([data['embedding'] for _, data in G_new.nodes(data=True)], dim=0)

    return G_new, edge_index.to(device), x.to(device)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, full_data: torch.Tensor):
        super(CustomDataset, self).__init__()
        self.data = full_data[:, 0:-1, :]
        self.targets = full_data[:, -1, 2].long()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        return sample, target
