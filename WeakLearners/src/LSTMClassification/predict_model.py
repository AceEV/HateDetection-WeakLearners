import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))).replace('\\', '/'))
sys.path.append(parent_dir)
print(parent_dir)
from src.LSTMClassification.LSTMClassification import LSTMClassification
from src.features.batches import Batches
import torch
from torch import nn
from datetime import datetime as dt
from config.paths import models_path
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from src.utility.load_params import load_params
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

params, params_dict = load_params(train=True)

pretrained_model_no = 110
pretrained_model_name = "run_teacher_training_all_layer_0.2_dropout_7"
pretrained_model_path = models_path + '/' + pretrained_model_name + '/model_{}.model'.format(pretrained_model_no)


class PredictModel:
	def __init__(self):
		self.end_key = None
		self.start_key = None
		self.writer = None

		self.model_path = models_path

		self.model = LSTMClassification(loss=nn.CrossEntropyLoss(), use_gpu=True, **params_dict)
		self.model.load_model(pretrained_model_path)

	def predict_model(self):
		batch = Batches(**params_dict)

		start_time = dt.now()
		self.model.network.eval()
		tloss = 0
		t_batches = 0
		yt = []
		yp = []
		for b in tqdm(batch.get_next_batch_test(), total=batch.test_batches_len):
			tweet_vec = torch.LongTensor(b['tweet_enc'].tolist())
			other_vec = torch.FloatTensor(b[['user_verified', 'user_followers_count', 'user_friends_count', 'retweet_count', 'fav_count']].values)
			y_true = torch.LongTensor(b['label'].tolist())

			_yp, _ = self.model.test(self.model.cuda(tweet_vec), self.model.cuda(other_vec), self.model.cuda(y_true))

			yt += y_true.unsqueeze(dim=1).cpu().detach().numpy().tolist()
			yp += torch.max(_yp, 1).indices.cpu().detach().numpy().tolist()
			tloss += self.model.step_loss.data.item()
			t_batches += 1


		end_time = dt.now()
		acc = accuracy_score(yt, yp)
		f1 = f1_score(yt, yp)
		confmat2 = confusion_matrix(y_true=yt, y_pred=yp)
		tloss /= t_batches

		print(recall_score(yt, yp))
		print(precision_score(yt, yp))

		print("Test")
		print('Loss:', tloss)
		print('Accuracy:', acc)
		print('F1 Score:', f1)
		print('Confusion Matrix\n', confmat2)
		print("---------------------Time taken: {}".format(end_time - start_time))


if __name__ == '__main__':
	PM = PredictModel()
	PM.predict_model()
