from utils import *
from Model_Operator import Model


class Predictor(nn.Module):
    def __init__(
            self,
            num_features_out: list = None,
            batch_size: int = 8,
            seq_length: int = 100,
            device=torch.device("cpu")
    ):
        super(Predictor, self).__init__()

        self.batch_size = batch_size
        self.train_loader = None
        self.validation_loader = None
        self.test_loader = None
        self.device = device

        self.Model = Model(
            num_features_out=num_features_out,
            batch_size=batch_size,
            seq_length=seq_length,
            device=device
        )

    def train(
            self,
            train_data: torch.Tensor = None,
            validation_data: torch.Tensor = None,
            poi_infos: list = None,
            epochs: int = 50,
            name: str = 'model.pt'
    ):
        train_set = CustomDataset(train_data)
        self.train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True
        )
        if validation_data is not None:
            validation_set = CustomDataset(validation_data)
            self.validation_loader = torch.utils.data.DataLoader(
                validation_set,
                batch_size=self.batch_size,
                shuffle=True
            )
        print("Start Training")
        for epoch in range(epochs):
            printProgressAction('Epoch ', epoch)
            # Train
            print("\n   Train")
            for num, (data_batch, targ_batch) in enumerate(self.train_loader):
                printProgressAction('     Batch', num)
                if data_batch.size(0) != self.batch_size:
                    continue
                self.Model.training_step(data_batch.to(self.device), poi_infos, targ_batch.to(self.device))
                # Clear unused memory
                torch.cuda.empty_cache()
            self.save(name=name)

            # Validation
            if epoch % 5 == 0 and validation_data is not None:
                print("\n   Validation")
                self.Model.pre_validation_step()
                for num, (data_batch, targ_batch) in enumerate(self.validation_loader):
                    printProgressAction('     Batch', num)
                    if data_batch.size(0) != self.batch_size:
                        continue
                    self.Model.validation_step(data_batch.to(self.device), poi_infos, targ_batch.to(self.device))
                    # Clear unused memory
                    torch.cuda.empty_cache()
                # EarlyStopping
                self.Model.stopping()
                if self.Model.done:
                    print(f"Stopped prematurely at iteration {epoch} due to EarlyStopping")
                    break
        print("\nEnd Training")

    def evaluation(
            self,
            test_data: torch.Tensor = None,
            poi_infos: list = None,
            k: list = None
    ):
        test_set = CustomDataset(test_data)
        self.test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=True
        )

        print("Start Evaluation")
        self.Model.pre_evaluation_step(dim=len(self.test_loader))
        for num, (data_batch, targ_batch) in enumerate(self.test_loader):
            printProgressAction('     Batch', num)
            if data_batch.size(0) != self.batch_size:
                continue
            self.Model.evaluation_step(data_batch.to(self.device), poi_infos, targ_batch.to(self.device), num)
            # Clear unused memory
            torch.cuda.empty_cache()

        # Final Evaluation Using Metrics
        print("\n        Results : ")
        for i in k:
            recall_i = recall_k(self.Model.eval_pred, self.Model.eval_targ, k=i)
            print(f"           recall_{i} : ", recall_i)
        print("End Evaluation")

    def save(self, name: str = 'model.pt'):
        torch.save(self.state_dict(), name)

    def load(self, name: str = 'model.pt'):
        self.load_state_dict(torch.load(name, map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
