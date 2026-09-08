from utils import *
from Predictor import Predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_set = upload_dataset("test_dataset")
pois_infos = [upload_dataset("tensor_poi_locations"), upload_dataset("tensor_poi_categories")]

pred = Predictor(num_features_out=[512, 785], batch_size=10, seq_length=99, device=device)
print("Untrained Model Evaluation : ")
pred.evaluation(test_data=test_set, poi_infos=pois_infos, k=[1, 5, 10])
pred.load(name='gnn_gat_st_lstm_model.pt')
print("Trained Model Evaluation : ")
pred.evaluation(test_data=test_set, poi_infos=pois_infos, k=[1, 5, 10])
