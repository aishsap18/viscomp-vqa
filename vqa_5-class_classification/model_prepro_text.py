import tensorflow as tf
import pandas as pd
import numpy as np
import os, h5py, sys, argparse
import ipdb
import time
import math
import cv2
import codecs, json
from tensorflow.compat.v1.nn import rnn_cell
from sklearn.metrics import average_precision_score
import gc
import sys
# from transformers import BertTokenizer, TFBertModel

class Answer_Generator():
	def __init__(self, dim_bert, rnn_size, rnn_layer, batch_size, input_embedding_size, dim_hidden, max_words_d, vocabulary_size_d, drop_out_rate, method):
		self.dim_bert = dim_bert
		self.rnn_size = rnn_size
		self.rnn_layer = rnn_layer
		self.batch_size = batch_size
		self.input_embedding_size = input_embedding_size
		self.dim_hidden = dim_hidden
		self.max_words_d = max_words_d
		self.vocabulary_size_d = vocabulary_size_d
		self.drop_out_rate = drop_out_rate
		self.method = method

		if self.method == 'lstm':
			# print("\nmethod: {}\n".format(self.method))
			# description-embedding
			self.embed_desc_W = tf.Variable(tf.random.uniform([self.vocabulary_size_d, self.input_embedding_size], -0.08, 0.08), name='embed_desc_W')

			# encoder: RNN body
			self.lstm_1_d = rnn_cell.LSTMCell(rnn_size, input_embedding_size, state_is_tuple=False)
			self.lstm_dropout_1_d = rnn_cell.DropoutWrapper(self.lstm_1_d, output_keep_prob = 1 - self.drop_out_rate)
			self.lstm_2_d = rnn_cell.LSTMCell(rnn_size, rnn_size, state_is_tuple=False)
			self.lstm_dropout_2_d = rnn_cell.DropoutWrapper(self.lstm_2_d, output_keep_prob = 1 - self.drop_out_rate)
			self.stacked_lstm_d = rnn_cell.MultiRNNCell([self.lstm_dropout_1_d, self.lstm_dropout_2_d])

			# description state-embedding
			self.embed_state_desc_W = tf.Variable(tf.random.uniform([2*rnn_size*rnn_layer, self.dim_hidden], -0.08,0.08),name='embed_state_desc_W')
		elif self.method == 'bert':
			# print("\nmethod: {}\n".format(self.method))
			# description state-embedding
			self.embed_state_desc_W = tf.Variable(tf.random.uniform([dim_bert, self.dim_hidden], -0.08,0.08),name='embed_state_desc_W')
		
		self.embed_state_desc_b = tf.Variable(tf.random.uniform([self.dim_hidden], -0.08, 0.08), name='embed_state_desc_b')

		# score embedding
		self.embed_score_W = tf.Variable(tf.random.uniform([dim_hidden, vocabulary_size_d], -0.08, 0.08), name='embed_score_W')
		self.embed_score_b = tf.Variable(tf.random.uniform([vocabulary_size_d], -0.08, 0.08), name='embed_score_b')

	def build_model(self):

		description = tf.compat.v1.placeholder(tf.int32, [self.batch_size, self.max_words_d])

		description_length = tf.compat.v1.placeholder(tf.int32, [self.batch_size])
		if self.method == 'lstm':
			# print("\nmethod: {}\n".format(self.method))
			# description embed
			state_d = self.stacked_lstm_d.zero_state(self.batch_size, dtype=tf.float32)
			for i in range(max_words_d):
				if i==0:
					desc_emb_linear = tf.zeros([self.batch_size, self.input_embedding_size])
				else:
					tf.get_variable_scope().reuse_variables()
					desc_emb_linear = tf.nn.embedding_lookup(self.embed_desc_W, description[:,i-1])

				desc_emb_drop = tf.nn.dropout(desc_emb_linear, 1-self.drop_out_rate)
				desc_emb = tf.tanh(desc_emb_drop)

				output_d, state_d = self.stacked_lstm_d(desc_emb, state_d)

			# description
			state_drop_d = tf.nn.dropout(state_d, 1-self.drop_out_rate)
			state_emb_d = tf.compat.v1.nn.xw_plus_b(tf.concat([state_drop_d[0], state_drop_d[1]], 1), self.embed_state_desc_W, self.embed_state_desc_b)
			# state_emb_d = tf.tanh(state_linear_d)

		elif self.method == 'bert':
			# print("\nmethod: {}\n".format(self.method))
			bert_description_embedding = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.dim_bert])

			state_drop_d = tf.nn.dropout(bert_description_embedding, 1-self.drop_out_rate)
			state_emb_d = tf.compat.v1.nn.xw_plus_b(state_drop_d, self.embed_state_desc_W, self.embed_state_desc_b)
			# state_emb_d = tf.tanh(state_linear_d)

		scores_emb = tf.nn.xw_plus_b(state_emb_d, self.embed_score_W, self.embed_score_b)

		description_list = tf.unstack(description)
		scores_emb_list = tf.unstack(scores_emb)
		description_length_list = tf.unstack(description_length)
		result = []
		result_desc = []
		for i in range(len(scores_emb_list)):
			tmp = tf.reshape(scores_emb_list[i], (1, self.vocabulary_size_d))
			repeats = [description_length_list[i], 1]
			result_desc.append(tf.slice(description_list[i], [0], [description_length_list[i]]))
			expanded_tensor = tf.expand_dims(tmp, -1)
			multiples = [1] + repeats
			tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
			tmp = tf.reshape(tiled_tensor, tf.shape(tmp) * repeats)
			result.append(tmp)
		
		scores_emb_repeated = tf.concat(result, axis=0)
		flatten_description = tf.concat(result_desc, axis=0)

		# Calculate cross entropy
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores_emb_repeated, labels=flatten_description)

		# mask = tf.math.logical_not(tf.math.is_nan(cross_entropy))
		# cross_entropy = tf.boolean_mask(cross_entropy, mask)
		loss = tf.reduce_mean(cross_entropy)
		
		if self.method == 'bert':
			# print("\nmethod: {}\n".format(self.method))
			return loss, description, state_emb_d, description_length, bert_description_embedding
		elif self.method == 'lstm':
			# print("\nmethod: {}\n".format(self.method))
			return loss, description, state_emb_d, description_length

	def build_generator(self):
		
		description = tf.compat.v1.placeholder(tf.int32, [self.batch_size, self.max_words_d])

		description_length = tf.compat.v1.placeholder(tf.int32, [self.batch_size])
		
		if self.method == 'lstm':
			# print("\nmethod: {}\n".format(self.method))
			# description embed
			state_d = self.stacked_lstm_d.zero_state(self.batch_size, dtype=tf.float32)
			for i in range(max_words_d):
				if i==0:
					desc_emb_linear = tf.zeros([self.batch_size, self.input_embedding_size])
				else:
					tf.get_variable_scope().reuse_variables()
					desc_emb_linear = tf.nn.embedding_lookup(self.embed_desc_W, description[:,i-1])

				desc_emb_drop = tf.nn.dropout(desc_emb_linear, 1-self.drop_out_rate)
				desc_emb = tf.tanh(desc_emb_drop)

				output_d, state_d = self.stacked_lstm_d(desc_emb, state_d)

			# description
			state_drop_d = tf.nn.dropout(state_d, 1-self.drop_out_rate)
			state_emb_d = tf.compat.v1.nn.xw_plus_b(tf.concat([state_drop_d[0], state_drop_d[1]], 1), self.embed_state_desc_W, self.embed_state_desc_b)
			# state_emb_d = tf.tanh(state_linear_d)

			return description, state_emb_d, description_length

		elif self.method == 'bert':
			# print("\nmethod: {}\n".format(self.method))
			bert_description_embedding = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.dim_bert])

			state_drop_d = tf.nn.dropout(bert_description_embedding, 1-self.drop_out_rate)
			state_emb_d = tf.compat.v1.nn.xw_plus_b(state_drop_d, self.embed_state_desc_W, self.embed_state_desc_b)
			# state_emb_d = tf.tanh(state_linear_d)
		
			return description, state_emb_d, description_length, bert_description_embedding



input_json = None #'./data_prepro_summary_qa_1700.json'
input_ques_h5 = None #'./data_prepro_summary_qa_1700.h5'
# out_name = None #'data_text_train_summary_qa_1700.h5'
input_bert_emb = None #'./data_text_bert_summary_qa_1700.h5'


# Train Parameters setting
learning_rate = 0.0003			# learning rate for rmsprop
#starter_learning_rate = 3e-4
learning_rate_decay_start = -1		# at what iteration to start decaying learning rate? (-1 = dont)
# batch_size = 500			# batch_size for each iterations
batch_size = 30
input_embedding_size = 200		# he encoding size of each token in the vocabulary
rnn_size = 512				# size of the rnn in number of hidden nodes in each layer
rnn_layer = 2				# number of the rnn layer
dim_hidden = 1024 #1024			# size of the common embedding vector
dim_bert = 768


decay_factor = 0.99997592083

# variation = ''

# Check point
checkpoint_path = ''

# misc
gpu_id = 0
# max_itr = 150000
max_itr = 1200  # 30000
# n_epochs = 1  # 3000
max_words_d = 50

def get_train_data(method, input_json, input_ques_h5, input_bert_emb=None):

	dataset = {}
	train_data = {}

	print('loading json file...')
	with open(input_json) as data_file:
		data = json.load(data_file)
	for key in data.keys():
		dataset[key] = data[key]

	vocabulary_size_d = len(dataset['ix_to_word_d'].keys())

	print('loading h5 file...')
	with h5py.File(input_ques_h5,'r') as hf:
		tem = hf.get('description_train')
		train_data['description'] = np.array(tem)-1
		
		# max length is 
		tem = hf.get('description_length_train')
		train_data['length_d'] = np.array(tem)

	print("\n\nmethod: {}\n\n".format(method))
	if method == 'bert':
		with h5py.File(input_bert_emb,'r') as hf:
			tem = hf.get('text_embeddings_train')
			train_data['text_embeddings'] = np.array(tem)

	return dataset, train_data


def train(input_json, input_ques_h5, input_bert_emb=None):
	
	print ('loading dataset...')
	dataset, train_data = get_train_data(method, input_json, input_ques_h5, input_bert_emb)

	num_train = train_data['description'].shape[0]

	desc_embed = np.empty((num_train, dim_hidden))

	vocabulary_size_d = len(dataset['ix_to_word_d'].keys())
	print ('vocabulary_size_d : ' + str(vocabulary_size_d))

	print ('constructing  model...')
	model = Answer_Generator(
			dim_bert = dim_bert,
			rnn_size = rnn_size,
			rnn_layer = rnn_layer,
			batch_size = batch_size,
			input_embedding_size = input_embedding_size,
			dim_hidden = dim_hidden,
			max_words_d = max_words_d,   
			vocabulary_size_d = vocabulary_size_d,  
			drop_out_rate = 0.5,
			method = method)

	if method == 'bert':
		# print("\nmethod: {}\n".format(method))
		tf_loss, tf_description, tf_embedding, tf_description_length, tf_bert_description_embedding = model.build_model()
	elif method == 'lstm':
		# print("\nmethod: {}\n".format(method))
		tf_loss, tf_description, tf_embedding, tf_description_length = model.build_model()

	sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
	saver = tf.train.Saver(max_to_keep=100)

	tvars = tf.compat.v1.trainable_variables()
	lr = tf.Variable(learning_rate)
	opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
	# gradient clipping
	gvs = opt.compute_gradients(tf_loss,tvars)
	clipped_gvs = [(tf.clip_by_value(grad, -10.0, 10.0), var) for grad, var in gvs]
	# tf.get_variable_scope().reuse_variables()
	# with tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:
	#     assert tf.get_variable_scope().reuse == True
	with tf.compat.v1.variable_scope('embed_ques_W/Adam', reuse=tf.compat.v1.AUTO_REUSE) as scope:
		train_op = opt.apply_gradients(clipped_gvs)

	# tf.initialize_all_variables().run()
	tf.compat.v1.global_variables_initializer().run()

	losses = []  #my code

	print ('start training...')
	tStart_total = time.time()

	for itr in range(max_itr+1):
		tStart = time.time()
		# shuffle the training data
		index = np.random.randint(0, num_train-1, batch_size)

		current_description = train_data['description'][index,:]
		current_length_d = train_data['length_d'][index]

		if method == 'bert':
			# print("\nmethod: {}\n".format(method))
			current_bert_description = train_data['text_embeddings'][index,:]

			_, loss, description, embedding, description_length, bert_desc = sess.run(
					[train_op, tf_loss, tf_description, tf_embedding, tf_description_length, tf_bert_description_embedding],
					feed_dict={
						tf_description: current_description,
						tf_description_length: current_length_d,
						tf_bert_description_embedding: current_bert_description
						})

		elif method == 'lstm':
			# print("\nmethod: {}\n".format(method))
			_, loss, description, embedding, description_length = sess.run(
					[train_op, tf_loss, tf_description, tf_embedding, tf_description_length],
					feed_dict={
						tf_description: current_description,
						tf_description_length: current_length_d
						})

		current_learning_rate = lr*decay_factor
		lr.assign(current_learning_rate).eval()

		tStop = time.time()
		if np.mod(itr, 100) == 0:
			print ("Iteration: ", itr, " Loss: ", loss, " Learning Rate: ", lr.eval())
			print ("Time Cost:", round(tStop - tStart,2), "s")

		if np.mod(itr, 400) == 0:
			print ("Iteration ", itr, " is done. Saving the model ...")
			saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step=itr)
			losses.append({'iteration': str(itr), 'loss': str(loss)})  


	print ("Finally, saving the model ...")
	saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step=max_itr)
	tStop_total = time.time()
	print ("Total Time Cost:", round(tStop_total - tStart_total,2), "s")
	sess.close() 
	gc.collect()

method = ''

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--method', default='lstm', help='enter method bert/lstm')
	parser.add_argument('--input_data_h5', required=True, help='enter input data h5 path')
	parser.add_argument('--input_data_json', required=True, help='enter input data json path')
	parser.add_argument('--input_bert_emb', default=None, help='enter input bert emb file path')
	parser.add_argument('--checkpoint_path', required=True, help='enter checkpoint path')

	args = parser.parse_args()
	params = vars(args)
	method = params['method']

	input_json = params['input_data_json']
	input_ques_h5 = params['input_data_h5']
	input_bert_emb = params['input_bert_emb']

	checkpoint_path = params['checkpoint_path']
	# "text_model_save/lstm/"

	with tf.device('/gpu:'+str(0)):
		train(input_json, input_ques_h5, input_bert_emb)
