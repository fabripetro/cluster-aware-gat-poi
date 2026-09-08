from utils import *
from Predictor import Predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = upload_dataset("train_dataset")
validation_set = upload_dataset("validation_dataset")
pois_infos = [upload_dataset("tensor_poi_locations"), upload_dataset("tensor_poi_categories")]

pred = Predictor(num_features_out=[512, 785], batch_size=10, seq_length=99, device=device)
pred.train(train_data=train_set, poi_infos=pois_infos, epochs=22, name='gnn_st_lstm_model.pt')
pred.save(name='gnn_gat_st_lstm_model.pt')
