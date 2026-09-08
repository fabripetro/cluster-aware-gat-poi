from utils import *
from Embedding import EmbeddingModule
from LSTM import SpatioTemporal_LSTM
from GraphConvolution import GNN
from GraphAttention import GATLayer
from NextPOICandidate import OutModule


class Generator(nn.Module):
    def __init__(
            self,
            num_features: int = 120,
            hidden_features: int = 64,
            embedding_features: int = 64,
            num_heads: int = 8,
            input_size_lstm: int = 220,
            hidden_size_lstm: int = 64,
            m_input_size: int = 60,
            num_layers_lstm: int = 1,
            out_features_out: int = 512,
            num_features_out: list = None,
            batch_size: int = 8,
            seq_length: int = 100,
            device=torch.device("cpu")
    ):
        super(Generator, self).__init__()
        self.batch_size = batch_size
        self.device = device

        self.embedding = EmbeddingModule(seq_length=seq_length, device=device)

        self.gnn = GNN(
            in_channels=num_features,
            hidden_channels=hidden_features,
            embedding_dim=embedding_features,
            device=device
        )
        self.gat = GATLayer(
            c_in=num_features,
            c_out=embedding_features,
            num_heads=num_heads,
            device=device
        )
        self.st_lstm = SpatioTemporal_LSTM(
            input_size=input_size_lstm,
            hidden_size=hidden_size_lstm,
            m_input_size=m_input_size,
            num_layers=num_layers_lstm,
            device=device
        )
        in_features_out = embedding_features + embedding_features + hidden_size_lstm
        self.output_module = OutModule(
            in_features_f=in_features_out,
            out_features_f=out_features_out,
            num_features=num_features_out,
            device=device
        )

    def __call__(
            self,
            curr_data: torch.Tensor,
            poi_infos: list,
            train_eval: bool = True
    ):
        next_poi = self.forward(curr_data, poi_infos, train_eval)
        return next_poi

    def forward(
            self,
            curr_data: torch.Tensor,
            poi_infos: list,
            train_eval: bool = True
    ):
        combined_embedding, poi_category_ids_embed, poi_position_embed, cluster_embed, candidate_poi_embed, diff_embed = self.embedding(curr_data, poi_infos)

        # General Pattern
        G_general, _, x_general = traj_graph(
            torch.cat([poi_category_ids_embed.unsqueeze(-2),
                       poi_position_embed.unsqueeze(-2),
                       cluster_embed.unsqueeze(-2)], dim=-2),
            k=1,
            cluster_discriminator=1,
            device=self.device)
        edge_list_general = []
        for u, v, _ in G_general.edges(keys=True):
            edge_list_general.append([u, v])
        edge_index_general = torch.tensor(edge_list_general, dtype=torch.long).t().contiguous()
        L = self.gnn(x_general.squeeze(), edge_index_general, train_eval=train_eval)

        # Local Pattern
        G_local, edge_index_local, x_local = traj_graph(
            torch.cat([poi_category_ids_embed.unsqueeze(-2),
                       poi_position_embed.unsqueeze(-2),
                       cluster_embed.unsqueeze(-2)], dim=-2),
            k=1,
            cluster_discriminator=0,
            device=self.device)
        adj_matrix = to_dense_adj(edge_index_local)[0].to(self.device)
        G = self.gat(x_local.unsqueeze(0), adj_matrix.unsqueeze(0))

        # Spatio-Temporal LSTM
        S_T_out, (S_T_h, S_T_c) = self.st_lstm(combined_embedding, diff_embed)

        cat_input = torch.cat([S_T_h.view(self.batch_size, -1),
                               G.repeat(self.batch_size, 1),
                               L.repeat(self.batch_size, 1)],
                              dim=-1)
        y = self.output_module(cat_input, candidate_poi_embed)

        return y
