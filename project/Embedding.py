import torch

from utils import *


class EmbeddingModule(nn.Module):
    def __init__(
            self,
            num_users: int = 3376,
            num_POI_ids: int = 100190,
            num_POI_category_ids: int = 785,
            num_clusters: int = 6,
            user_embedding_dim: int = 10,
            poi_ids_embedding_dim: int = 20,
            poi_category_ids_embedding_dim: int = 60,
            poi_position_embedding_dim: int = 60,
            checkin_time_embedding_dim: int = 10,
            cluster_embedding_dim: int = 60,
            candidate_poi_embedding_dim: int = 30,
            spatio_temporal_diff_embedding_dim: int = 60,
            current_traj_dim: int = 1,
            seq_length: int = 100,
            negative_slope: float = 0.01,
            device=torch.device("cpu")
    ):
        super(EmbeddingModule, self).__init__()
        self.m = current_traj_dim
        self.device = device

        # Embedding Layers
        self.user_embedding = nn.Embedding(num_users, user_embedding_dim, device=device)
        self.poi_ids_embedding = nn.Embedding(num_POI_ids, poi_ids_embedding_dim, device=device)
        self.poi_category_ids_embedding = nn.Embedding(num_POI_category_ids, poi_category_ids_embedding_dim, device=device)
        self.poi_position_embedding = nn.Linear(1, poi_position_embedding_dim, device=device)
        self.checkin_time_embedding = nn.Linear(1, checkin_time_embedding_dim, device=device)
        self.cluster_embedding = nn.Embedding(num_clusters, cluster_embedding_dim, device=device)
        self.candidate_poi_embedding = nn.Embedding(num_POI_category_ids, candidate_poi_embedding_dim, device=device)

        # Spatio-Temporal Interval Embedding
        self.diff_embedding = nn.Linear(2, spatio_temporal_diff_embedding_dim, device=device)

        # Feature Fusion
        dim = (user_embedding_dim + poi_ids_embedding_dim + poi_category_ids_embedding_dim +
               poi_position_embedding_dim + checkin_time_embedding_dim + cluster_embedding_dim)
        self.fusion = nn.Sequential(
            nn.Linear(
                in_features=dim,
                out_features=dim,
                device=device
            ),
            nn.BatchNorm1d(num_features=seq_length, device=device),
            nn.LeakyReLU(negative_slope=negative_slope)
        )

    def __call__(
            self,
            x: torch.Tensor,
            poi_infos: list
    ):
        combined_embedding = self.forward(x, poi_infos)
        return combined_embedding

    def spatio_temporal_difference(
            self,
            poi_position: torch.Tensor,
            checkin_time: torch.Tensor,
            epsilon: float = 1e-10
    ):
        batch_size, seq_length = poi_position.size()
        difference = torch.zeros([batch_size, seq_length, 2], device=self.device)
        for i in range(batch_size):
            spatial_range = torch.max(poi_position[i, :]) - torch.min(poi_position[i, :]) + epsilon
            temporal_range = torch.max(checkin_time[i, :]) - torch.min(checkin_time[i, :]) + epsilon
            for j in range(seq_length-1):
                if poi_position[i, j] != 0 and checkin_time[i, j] != 0:
                    difference[i, j, 0] = torch.abs(poi_position[i, j] - poi_position[i, j+1]) / spatial_range
                    difference[i, j, 1] = torch.abs(checkin_time[i, j] - checkin_time[i, j+1]) / temporal_range
                else:
                    break
            difference[i, seq_length-1, 0] = torch.abs(poi_position[i, seq_length-1] - 0) / spatial_range
            difference[i, seq_length-1, 1] = torch.abs(checkin_time[i, seq_length-1] - 0) / temporal_range
        return difference

    @staticmethod
    def find_closest_POIs(
            current_positions: torch.Tensor,
            current_categories: torch.Tensor,
            poi_positions: torch.Tensor,
            poi_categories: torch.Tensor
    ):
        B, N = current_positions.shape[0], poi_positions.shape[0]

        distances = torch.abs(current_positions.unsqueeze(0) - poi_positions.unsqueeze(1)).T
        mask = poi_categories.unsqueeze(0).expand(B, N) == current_categories.unsqueeze(1).expand(B, N)
        distances.masked_fill_(mask, float('inf'))
        closest_poi_indices = torch.argmin(distances, dim=1)
        closest_poi_categories = poi_categories[closest_poi_indices]

        return closest_poi_categories.int()

    def forward(self, x: torch.Tensor, poi_infos: list):
        # x : [batch_size, seq_length, 6]
        user_ids = x[:, :, 0].int()    # int
        poi_ids = x[:, :, 1].int()  # int
        poi_category_ids = x[:, :, 2].int()    # int
        poi_position = x[:, :, 3].float()    # float
        checkin_time = x[:, :, 4].float()    # float
        clusters = x[:, :, 5].int()    # int
        # poi_infos : [[num_POIs, 2], [num_POI_categories]]
        poi_positions, poi_categories = poi_infos[0][:, 1].to(self.device), poi_infos[1][:, 1].to(self.device)

        # Trajectory Embeddings
        user_embed = self.user_embedding(user_ids)  # [batch_size, seq_length, embedding_dim]
        poi_ids_embed = self.poi_ids_embedding(poi_ids)  # [batch_size, seq_length, embedding_dim]
        poi_category_ids_embed = self.poi_category_ids_embedding(poi_category_ids)   # [batch_size, seq_length, embedding_dim]
        poi_position_embed = self.poi_position_embedding(poi_position.unsqueeze(-1))  # [batch_size, seq_length, embedding_dim]
        checkin_time_embed = self.checkin_time_embedding(checkin_time.unsqueeze(-1))  # [batch_size, seq_length, embedding_dim]
        cluster_embed = self.cluster_embedding(clusters)    # [batch_size, seq_length, embedding_dim]

        # Spatio-Temporal Interval Embeddings
        diff = self.spatio_temporal_difference(poi_position, checkin_time)
        diff_embed = self.diff_embedding(diff)  # [batch_size, seq_length, 1, embedding_dim]

        # Candidate POI Embeddings
        candidate_poi = self.find_closest_POIs(
            current_positions=poi_position[:, -1],
            current_categories=poi_category_ids[:, -1],
            poi_positions=poi_positions,
            poi_categories=poi_categories)
        candidate_poi_embed = self.candidate_poi_embedding(candidate_poi)

        # Concatenate all embedding
        combined_embedding = torch.cat(
            [user_embed, poi_ids_embed, poi_category_ids_embed, poi_position_embed, checkin_time_embed, cluster_embed],
            dim=-1
        ).squeeze()   # [batch_size, seq_length, 6*embedding_dims]
        combined_embedding = self.fusion(combined_embedding)

        return combined_embedding, poi_category_ids_embed, poi_position_embed, cluster_embed, candidate_poi_embed, diff_embed
