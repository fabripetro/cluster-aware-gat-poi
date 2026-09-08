from utils import *


class GATLayer(nn.Module):
    def __init__(
            self,
            c_in: int,
            c_out: int,
            num_heads: int = 1,
            concat_heads: bool = True,
            alpha: float = 0.2,
            device=torch.device("cpu")
    ):
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % num_heads == 0
            c_out = c_out // num_heads

        # Submodules and parameters needed in the layer
        self.projection = nn.Linear(c_in, c_out * num_heads, device=device)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * c_out)).to(device)    # One per head
        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(
            self,
            node_feats,
            adj_matrix,
            batch=None,
    ):
        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)

        # Project input to multi-head space
        node_feats = self.projection(node_feats)
        node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)

        # Get edge indices
        edges = adj_matrix.nonzero(as_tuple=False)
        node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)
        edge_indices_row = edges[:, 0] * num_nodes + edges[:, 1]
        edge_indices_col = edges[:, 0] * num_nodes + edges[:, 2]

        # Concatenate features of edge endpoints
        a_input = torch.cat([
            torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0),
            torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0)
        ], dim=-1)

        attn_logits = torch.einsum('bhc,hc->bh', a_input, self.a)
        attn_logits = self.leaky_relu(attn_logits)

        # Build masked attention matrix
        attn_matrix = attn_logits.new_zeros(adj_matrix.shape+(self.num_heads,)).fill_(-9e15)
        attn_matrix[adj_matrix[..., None].repeat(1, 1, 1, self.num_heads) == 1] = attn_logits.reshape(-1)

        attn_probs = F.softmax(attn_matrix, dim=2)
        if attn_probs.size(1) != node_feats.size(1):
            new_shape = list(attn_probs.shape)
            new_shape[1] = node_feats.size(1)
            new_shape[2] = new_shape[1]
            attn_probs = attn_probs.new_zeros(new_shape)
        node_feats = torch.einsum('bijh,bjhc->bihc', attn_probs, node_feats)

        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1).squeeze()
        else:
            node_feats = node_feats.mean(dim=2)

        if batch is not None:
            node_feats = global_mean_pool(node_feats, batch)
        else:
            node_feats = node_feats.mean(dim=0, keepdim=True)

        return node_feats
