from utils import *
from Generator import Generator


class Model(nn.Module):
    def __init__(
            self,
            batch_size: int = 8,
            seq_length: int = 100,
            num_features_out: list = None,
            learning_rate: float = 3e-4,
            weight_decay: float = 1e-5,
            device=torch.device("cpu")
    ):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.device = device

        self.loss = nn.CrossEntropyLoss()

        self.validation_losses = []
        self.previous_validation_losses = self.validation_losses
        self.count = 0
        self.done = False
        self.eval_pred, self.eval_targ = None, None

        # Model
        self.predictor = Generator(
            num_layers_lstm=3,
            num_features_out=num_features_out,
            batch_size=batch_size,
            seq_length=seq_length,
            device=device
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.predictor.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Initialization
        self.predictor.apply(init_weights)

    def training_step(
            self,
            x: torch.Tensor,
            poi_infos: list,
            targ: torch.Tensor,
            train_eval: bool = True
    ):
        pred_next_poi = self.predictor(x, poi_infos, train_eval)
        self.update_step(pred_next_poi, targ)

    def update_step(
            self,
            pred_next_poi: torch.Tensor,
            targ: torch.Tensor
    ):
        self.optimizer.zero_grad()
        loss = self.loss(pred_next_poi, targ)
        loss.backward()
        self.optimizer.step()

    def pre_validation_step(self):
        self.validation_losses = []
        self.previous_validation_losses = self.validation_losses
        self.count = 0

    def validation_step(
            self,
            x: torch.Tensor,
            poi_infos: list,
            targ: torch.Tensor,
            train_eval: bool = False
    ):
        pred_next_poi = self.predictor(x, poi_infos, train_eval)
        self.post_validation_step(pred_next_poi, targ)

    def post_validation_step(
            self,
            pred_next_poi: torch.Tensor,
            targ: torch.Tensor
    ):
        loss = self.loss(pred_next_poi, targ)
        self.validation_losses.append(loss)

    def pre_evaluation_step(
            self,
            dim: int = 1,
            num_categories: int = 785,
    ):
        self.eval_pred = torch.zeros([dim, self.batch_size, num_categories])
        self.eval_targ = torch.zeros([dim, self.batch_size])

    def evaluation_step(
            self,
            x: torch.Tensor,
            poi_infos: list,
            targ: torch.Tensor,
            curr_num: int,
            train_eval: bool = False
    ):
        pred_next_poi = self.predictor(x, poi_infos, train_eval)
        self.eval_pred[curr_num, :, :] = F.softmax(pred_next_poi, dim=-1)
        self.eval_targ[curr_num, :] = targ

    def stopping(self):
        if self.previous_validation_losses is None:
            self.count = 0
        else:
            self.count, self.done = EarlyStopping(statistics.mean(self.validation_losses),
                                                  statistics.mean(self.previous_validation_losses),
                                                  self.count)
