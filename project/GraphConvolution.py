from utils import *


class GNN(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            embedding_dim: int,
            device=torch.device("cpu")
    ):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, device=device)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, device=device)
        self.conv3 = GCNConv(hidden_channels, embedding_dim, device=device)

    def __call__(self, x, edge_index, batch=None, train_eval=True):
        y = self.forward(x, edge_index, batch)
        return y

    def forward(
            self,
            x,
            edge_index,
            batch=None,
            train_eval=True
    ):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = F.dropout(x, p=0.5, training=train_eval)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = F.dropout(x, p=0.5, training=train_eval)
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = F.dropout(x, p=0.5, training=train_eval)

        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)

        return x


class GCNConv(MessagePassing):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            device=torch.device("cpu")
    ):
        super(GCNConv, self).__init__(aggr='add')
        self.device = device

        self.lin = Linear(in_channels, out_channels, bias=False, device=device)
        self.bias = Parameter(torch.empty(out_channels)).to(device)

        self.reset_parameters()

    def __call__(
            self,
            x,
            edge_index
    ):
        y = self.forward(x, edge_index)
        return y

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(
            self,
            x,
            edge_index
    ):
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Apply a linear transformation to node features
        x = self.lin(x)

        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Propagating messages
        out = self.propagate(edge_index, x=x, norm=norm)

        # Add bias after aggregation
        out = out + self.bias

        return out

    def message(self, x_j, norm, edge_attr=None):
        msg = norm.view(-1, 1).to(self.device) * x_j
        if edge_attr is not None:
            msg = msg * edge_attr.view(-1, 1)
        return msg
