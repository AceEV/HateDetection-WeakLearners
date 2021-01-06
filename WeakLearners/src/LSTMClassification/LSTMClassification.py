import torch
import torch.nn as nn
from torch.optim import Adam
from src.utility.common_functions import create_parent_dir, write_json

torch.manual_seed(1)


class Network(nn.Module):
	def __init__(self, **kwargs):
		super(Network, self).__init__()

		self.tweet_embd = nn.Embedding(kwargs["n_vocab"], kwargs["n_embd"])

		self.rnn = nn.LSTM(input_size=kwargs["n_embd"], hidden_size=kwargs["n_rnn_hidden"],
						   batch_first=True, num_layers=2, dropout=kwargs["dropout"])
		self.linear = nn.Sequential(nn.Dropout(kwargs["dropout"]),
									nn.Linear(kwargs["n_rnn_hidden"], kwargs["n_ann_hidden"]), nn.ReLU(),
									nn.Linear(kwargs["n_ann_hidden"], kwargs["n_out"]))

	def forward(self, tweet, other_info, states=None):
		tw_embd = self.tweet_embd(tweet)

		if not states:
			rnn_out, states = self.rnn(tw_embd)
		else:
			rnn_out, states = self.rnn(tw_embd, states)
		y_out = self.linear(rnn_out[:, -1, :])
		return y_out, states


class Network2(nn.Module):
	def __init__(self, **kwargs):
		super(Network2, self).__init__()

		self.tweet_embd = nn.Embedding(kwargs["n_vocab"], kwargs["n_embd"])

		self.rnn = nn.LSTM(input_size=kwargs["n_embd"], hidden_size=kwargs["n_rnn_hidden"],
						   batch_first=True, num_layers=2, dropout=kwargs["dropout"])
		self.linear = nn.Sequential(nn.Dropout(kwargs["dropout"]),
									nn.Linear(kwargs["n_rnn_hidden"]+kwargs['n_other_info'], kwargs["n_ann_hidden"]), nn.ReLU(),
									nn.Linear(kwargs["n_ann_hidden"], kwargs["n_ann_hidden"]), nn.ReLU(),
									nn.Linear(kwargs["n_ann_hidden"], kwargs["n_out"]))

	def forward(self, tweet, other_info, states=None):
		tw_embd = self.tweet_embd(tweet)

		if not states:
			rnn_out, states = self.rnn(tw_embd)
		else:
			rnn_out, states = self.rnn(tw_embd, states)
		d = torch.cat((rnn_out[:, -1, :], other_info), dim=1)
		y_out = self.linear(d)
		return y_out, states


class LSTMClassification():
	def __init__(self, optim=Adam, loss=nn.MSELoss(), use_gpu=False, **kwargs):
		super(LSTMClassification, self).__init__()
		self.use_gpu = use_gpu
		self.cuda = lambda x: x.cuda(kwargs['device_id']) if (torch.cuda.is_available() and use_gpu) else x
		self.network = self.cuda(Network2(**kwargs))
		self.loss = loss
		self.kwargs = kwargs
		self.optim = optim(self.network.parameters(), lr=kwargs['learning_rate'])
		self.step_loss = None
		self.y_pred = None

	def evaluate(self, **kwargs):
		pass

	def save_model1(self, path):
		create_parent_dir(path)
		self.optim.zero_grad()
		torch.save(self.network.cpu().state_dict(), path)

	def save_params(self, path, params):
		create_parent_dir(path)
		write_json(params, path)

	def load_model(self, path):
		self.network.load_state_dict(torch.load(path), strict=False)
		self.network.eval()

	def train(self, x1, x2, y_true):
		self.optim.zero_grad()
		y_pred, states = self.network(x1, x2)
		self.step_loss = self.loss(torch.reshape(y_pred, (-1, self.kwargs['n_out'])), torch.reshape(y_true, (-1,)))
		self.y_pred = y_pred
		# self.step_loss = self.loss(
		# 	torch.reshape(self.y_pred, (-1, _n_sent_embd)), torch.reshape(y_true, (-1, _n_sent_embd)))
		self.step_loss.backward()
		self.optim.step()

	def test(self, x1, x2, y_true):
		y_pred, states = self.network(x1, x2)
		self.step_loss = self.loss(torch.reshape(y_pred, (-1, self.kwargs['n_out'])), torch.reshape(y_true, (-1,)))
		return y_pred, states

	def predict(self, x1, x2, x3, states):
		y_pred, states = self.network(x1, x2, x3, states)
		return y_pred, states


if __name__ == '__main__':
	params = {
		"run_no": 1,
		"n_epochs": 500,
		"max_batch_size": 16,
		"learning_rate": 0.001,
		"dropout": 0.2,
		"n_vocab": 1000,
		"n_embd": 128,
		"n_rnn_hidden": 128,
		"n_out": 3,
		"n_ann_hidden": 64,
		"end_tensor_key": 1,
		"start_tensor_key": 0,
		"device_id": 0
	}
	_n_batch = 5
	x1 = torch.randint(0, 1000, (_n_batch, 5))
	y_true = torch.randint(0, 3, (_n_batch,))

	model = LSTMClassification(loss=nn.CrossEntropyLoss(), **params)
	model.train(x1, y_true)

	print(model.network(x1)[0].shape)
