import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))).replace('\\', '/'))
sys.path.append(parent_dir)
print(parent_dir)
from src.LSTMClassification.LSTMClassification import LSTMClassification
from src.features.batches import Batches
import torch
from src.utility.common_functions import create_dir
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime as dt
from config.paths import models_path, summary_path
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from src.utility.load_params import load_params
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

params, params_dict = load_params(train=True)


class TrainModel:
	def __init__(self):
		self.model = LSTMClassification(loss=nn.CrossEntropyLoss(), use_gpu=True, **params_dict)
		self.end_key = None
		self.start_key = None
		self.writer = None

		self.model_path = models_path
		self.create_summary()

	def create_summary(self):
		summ_path = summary_path
		create_dir(summ_path)
		create_dir(self.model_path)
		self.writer = SummaryWriter(summ_path + '/' + params.model_name)

	def train_model(self):
		# if does_dir_exist(self.model_path + '/' + params.model_name):
		# 	print("Model already exists!!!")
		# 	sys.exit()

		# self.model.network = torch.nn.DataParallel(self.model.network, device_ids=[4, 5, 6, 7])

		print("-------------------------------------------------------")
		print("Run number: {}".format(params_dict['run_no']))
		print("-------------------------------------------------------""")

		batch = Batches(**params_dict)

		label_name = 'label'
		if params_dict['type'] == 'weak':
			label_name = 'snorkel_label'

		print("START TRAINING.........................")
		start_time = dt.now()
		for epoch in range(1, params.n_epochs+1):
			self.model.network.train()
			print("Epoch: ", epoch)
			tloss = 0
			t_batches = 0
			yt = []
			yp = []
			for b in tqdm(batch.get_next_batch_train(), total=batch.train_batches_len):
				tweet_vec = torch.LongTensor(b['tweet_enc'].tolist())
				other_vec = torch.FloatTensor(b[['user_verified', 'user_followers_count', 'user_friends_count', 'retweet_count', 'fav_count']].values)
				y_true = torch.LongTensor(b[label_name].tolist())

				self.model.train(self.model.cuda(tweet_vec), self.model.cuda(other_vec), self.model.cuda(y_true))

				yt += y_true.unsqueeze(dim=1).cpu().detach().numpy().tolist()
				yp += torch.max(self.model.y_pred, 1).indices.cpu().detach().numpy().tolist()
				tloss += self.model.step_loss.data.item()
				t_batches += 1

			end_time = dt.now()
			acc = accuracy_score(yt, yp)
			f1 = f1_score(yt, yp)
			confmat2 = confusion_matrix(y_true=yt, y_pred=yp)
			tloss /= t_batches


			print("Train")
			print('Loss:', tloss)
			print('Accuracy:', acc)
			print('F1 Score:', f1)
			print('Confusion Matrix\n', confmat2)
			print("---------------------Time taken: {}".format(end_time - start_time))

			self.writer.add_scalar("tot_loss", tloss, global_step=epoch)
			self.writer.add_scalar("Accuracy", acc, global_step=epoch)
			self.writer.add_scalar("F1 Score", f1, global_step=epoch)

			start_time = end_time
			if epoch % 5 == 0:
				self.model.save_model1(self.model_path + '/' + params.model_name + '/model_{}.model'.format(epoch))
				self.model.save_params(self.model_path + '/' + params.model_name + '/params.json', params_dict)
				self.test_model(batch, epoch, label_name)

	def test_model(self, batch, epoch, label_name):
		tloss = 0
		t_batches = 0
		yt = []
		yp = []
		self.model.network.eval()
		start_time = dt.now()

		for b in tqdm(batch.get_next_batch_test(), total=batch.test_batches_len):

			tweet_vec = torch.LongTensor(b['tweet_enc'].tolist())
			other_vec = torch.FloatTensor(b[['user_verified', 'user_followers_count', 'user_friends_count', 'retweet_count', 'fav_count']].values)
			y_true = torch.LongTensor(b[label_name].tolist())

			_yp, _ = self.model.test(self.model.cuda(tweet_vec), self.model.cuda(other_vec), self.model.cuda(y_true))

			yt += y_true.unsqueeze(dim=1).cpu().detach().numpy().tolist()
			yp += torch.max(_yp, 1).indices.cpu().detach().numpy().tolist()
			tloss += self.model.step_loss.data.item()
			t_batches += 1

		if t_batches == 0:
			t_batches = 1

		end_time = dt.now()
		acc = accuracy_score(yt, yp)
		f1 = f1_score(yt, yp)
		confmat2 = confusion_matrix(y_true=yt, y_pred=yp)
		tloss /= t_batches
		print()
		print("*" * 60)
		print("Test")
		print('Loss:', tloss)
		print('Accuracy:', acc)
		print('F1 Score:', f1)
		print('Confusion Matrix\n', confmat2)
		print("Time taken: {}".format(end_time - start_time))
		print("*" * 60)

		self.writer.add_scalar("tot_loss_test", tloss, global_step=epoch)
		self.writer.add_scalar("Acc_test", acc, global_step=epoch)
		self.writer.add_scalar("F1_test", f1, global_step=epoch)


if __name__ == '__main__':
	TM = TrainModel()
	TM.train_model()
