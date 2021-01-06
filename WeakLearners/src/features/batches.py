from sklearn.model_selection import train_test_split
import pandas as pd
from tokenizers import Tokenizer
from config.paths import data_path, embeddings_path
from src.utility.common_functions import timing
import random

pd.set_option('display.max_columns', None)

random_state = 1
_vocab_path = embeddings_path + '/wordpiece2.txt'

targets = ['racism', 'sexism', 'neither']

label_map = {
	"nonhate": 0,
	"hate": 1
}

reverse_label_map = {
	0: "nonhate",
	1: "hate"
}
#
# label_map = {
# 	"racism": 0,
# 	"sexism": 1,
# 	"neither": 2
# }
#
# reverse_label_map = {
# 	0: "racism",
# 	1: "sexism",
# 	2: "neither"
# }

user_verified_map = {
	True: 1,
	False: 0
}


class Batches:
	def __init__(self, **kwargs):
		self.max_batch_size = kwargs['max_batch_size']
		if kwargs['type'] == 'weak':
			filename = 'weak_data.csv'
		else:
			filename = 'labelled_data.csv'

		self.data_path = data_path + '/' + filename

		self.all_data = None
		self.train_data = None
		self.test_data = None
		self.train_batches_len = 0
		self.test_batches_len = 0
		self.tokenizer = Tokenizer.from_file(_vocab_path)
		self.load_all_data()
		self.create_train_test()

	def normalize(self, column):
		# self.all_data[column] = (self.all_data[column]-self.all_data[column].mean()) / self.all_data[column].std()
		self.all_data[column] = self.all_data[column] / (self.all_data[column].max() - self.all_data[column].min())

	def encode_tweet(self, tweet):
		# tweet = "<start>" + tweet + "<end>"
		return self.tokenizer.encode(tweet, add_special_tokens=False).ids

	@timing
	def load_all_data(self):
		self.all_data = pd.read_csv(self.data_path, usecols=['id', 'text', 'user', 'user_verified',
			'user_followers_count', 'user_friends_count', 'retweet_count', 'fav_count', 'target', 'label', 'snorkel_label'])[:]
		self.all_data.dropna(inplace=True)
		# print(self.all_data.info())
		# self.all_data['processed_target'] = self.all_data.processed_target.apply(lambda x: 'neither' if x == 'none' else x)
		# self.all_data['processed_target'] = self.all_data.target.apply(lambda x: x.split()[1])
		# self.all_data = self.all_data[self.all_data['processed_target'].isin(targets)]
		# self.all_data['label'] = self.all_data.processed_target.apply(lambda x: label_map[x])

		self.all_data['tweet_enc'] = self.all_data['text'].apply(self.encode_tweet)
		self.all_data['tweet_enc_len'] = self.all_data['tweet_enc'].apply(lambda x: len(x))
		# print(self.all_data['tweet_enc_len'].value_counts())

		self.all_data['user_verified'] = self.all_data.user_verified.apply(lambda x: user_verified_map[x])

		self.normalize('user_followers_count')
		self.normalize('user_friends_count')
		self.normalize('retweet_count')
		self.normalize('fav_count')

		self.all_data.dropna(inplace=True)

		print("Data Loaded!!")

	def create_train_test(self):
		self.train_data, self.test_data = train_test_split(self.all_data, test_size=0.2, random_state=random_state, stratify=self.all_data['label'])
		for i, grp in self.train_data.groupby("tweet_enc_len"):
			grpbat = len(grp) // self.max_batch_size
			grpbat = grpbat + 1 if len(grp) % self.max_batch_size != 0 else grpbat
			self.train_batches_len += grpbat
		for i, grp in self.test_data.groupby("tweet_enc_len"):
			grpbat = len(grp) // self.max_batch_size
			grpbat = grpbat + 1 if len(grp) % self.max_batch_size != 0 else grpbat
			self.test_batches_len += grpbat

	def get_next_batch(self, df):
		grouped = df.groupby("tweet_enc_len")
		group_keys = list(grouped.groups.keys())
		random.shuffle(group_keys)
		for name in group_keys:
			group = grouped.get_group(name)
			for i in range(0, len(group), self.max_batch_size):
				yield group[i:i+self.max_batch_size].reset_index()

	def get_next_batch_train(self):
		return self.get_next_batch(self.train_data)

	def get_next_batch_test(self):
		return self.get_next_batch(self.test_data)

	def __len_train__(self):
		return


if __name__ == '__main__':
	kwargs = {
		'max_batch_size': 32,
		'type': "weak"
	}

	action = 'train'
	batches = Batches(**kwargs)

	cnt = 0

	for b in batches.get_next_batch_train():
		# print(b.tweet_enc)
		# print(b.user_followers_count)
		# print(b.user_friends_count)
		# print(b.tweet_enc_len)
		# print(batches.train_batches_len)
		# print("*"*12)
		# break
		cnt += 1
	print(cnt)
