import torch

from utils import *


class OutModule(nn.Module):
    def __init__(
            self,
            in_features_f: int,
            out_features_f: int,
            out_features_c_a: int = 128,
            num_features: list = None,
            num_POI_classes: int = 785,
            negative_slope: float = 0.01,
            device=torch.device("cpu")
    ):
        super(OutModule, self).__init__()
        self.device = device

        self.fusion = nn.Sequential(
            nn.Linear(
                in_features=in_features_f,
                out_features=out_features_f,
                device=device
            ),
            nn.BatchNorm1d(num_features=num_features[0], device=device),
            nn.LeakyReLU(negative_slope=negative_slope)
        )

        self.cross_attention = CrossAttention(
            out_features_q=out_features_c_a,
            out_features_k=out_features_c_a,
            out_features_v=out_features_c_a,
            device=device
        )

        self.classification = nn.Sequential(
            nn.Linear(
                in_features=out_features_c_a,
                out_features=num_POI_classes,
                device=device
            ),
            nn.BatchNorm1d(num_features=num_features[1], device=device),
            nn.LeakyReLU(negative_slope=negative_slope)
        )

    def __call__(
            self,
            cat_input: torch.Tensor,
            D: torch.Tensor,
    ):
        y = self.forward(cat_input, D)
        return y

    def forward(
            self,
            cat_input: torch.Tensor,
            D: torch.Tensor,
    ):
        fused_input = self.fusion(cat_input)

        attention_output = self.cross_attention(fused_input, D)

        return self.classification(attention_output)


class CrossAttention(nn.Module):
    def __init__(
            self,
            in_features_q: int = 30,
            out_features_q: int = 128,
            in_features_k: int = 512,
            out_features_k: int = 128,
            in_features_v: int = 512,
            out_features_v: int = 128,
            device=torch.device("cpu")
    ):
        super(CrossAttention, self).__init__()
        self.device = device

        self.Q = nn.Linear(
            in_features=in_features_q,
            out_features=out_features_q,
            bias=False,
            device=device
        )
        self.K = nn.Linear(
            in_features=in_features_k,
            out_features=out_features_k,
            bias=False,
            device=device
        )
        self.V = nn.Linear(
            in_features=in_features_v,
            out_features=out_features_v,
            bias=False,
            device=device
        )
        self.softmax = nn.Softmax(dim=-1)

    def __call__(self, fused_features: torch.Tensor, D: torch.Tensor):
        y = self.forward(fused_features, D)
        return y

    def forward(self, fused_features: torch.Tensor, D: torch.Tensor):
        query = self.Q(D)
        key = self.K(fused_features)
        value = self.V(fused_features)

        d = query.size(-1)
        Q_K = query @ torch.transpose(key, 0, 1)

        y = self.softmax((Q_K / (d ** 0.5))) @ value

        return y
