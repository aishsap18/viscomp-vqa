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
from model_prepro_text import *

def get_test_data():

	dataset = {}
	test_data = {}

	print('loading json file...')
	with open(input_json) as data_file:
		data = json.load(data_file)
	for key in data.keys():
		dataset[key] = data[key]

	vocabulary_size_d = len(dataset['ix_to_word_d'].keys())

	print('loading h5 file...')
	with h5py.File(input_ques_h5,'r') as hf:
		tem = hf.get('description_test')
		test_data['description'] = np.array(tem)-1
        # max length is 
		tem = hf.get('description_length_test')
		test_data['length_d'] = np.array(tem)
		# print("test: {}".format(len(test_data['length_d'])))

	if method == 'bert':
		# print("\nmethod: {}\n".format(method))
		with h5py.File(input_bert_emb,'r') as hf:
			tem = hf.get('text_embeddings_test')
			test_data['text_embeddings'] = np.array(tem)

	return dataset, test_data


def test(model_path, dataset, data_train, data_test):
	# print ('loading dataset...')
	# dataset, test_data = get_test_data()

	vocabulary_size_d = len(dataset['ix_to_word_d'].keys())

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
		tf_description, tf_embedding, tf_description_length, tf_bert_description_embedding = model.build_generator()
	elif method == 'lstm':
		# print("\nmethod: {}\n".format(method))
		tf_description, tf_embedding, tf_description_length = model.build_generator()

	sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
	saver = tf.train.Saver()
	saver.restore(sess, model_path)

	def get_embeddings(num, data):
		
		desc_embed = np.empty((num, dim_hidden))
		# print("\n\nkeys: {}\n\n".format(data.keys()))
		
		tStart_total = time.time()
		result = []
		for current_batch_start_idx in range(0,num,batch_size):
			tStart = time.time()

			if current_batch_start_idx + batch_size < num:
				current_batch_file_idx = range(current_batch_start_idx,current_batch_start_idx+batch_size)
			else:
				current_batch_file_idx = range(current_batch_start_idx,num)

			current_description = data['description'][current_batch_file_idx,:]
			current_length_d = data['length_d'][current_batch_file_idx]

			if method == 'bert':
				# print("\nmethod: {}\n".format(method))
				current_bert_description = data['text_embeddings'][current_batch_file_idx,:]

			if(len(current_description)<batch_size):
				pad_d = np.zeros((batch_size-len(current_description),max_words_d),dtype=np.int)
				pad_d_len = np.zeros(batch_size-len(current_length_d),dtype=np.int)

				current_description = np.concatenate((current_description, pad_d))
				current_length_d = np.concatenate((current_length_d, pad_d_len))

				if method == 'bert':
					# print("\nmethod: {}\n".format(method))
					pad_d_emb = np.zeros((batch_size-len(current_bert_description),dim_bert),dtype=np.float)
					current_bert_description = np.concatenate((current_bert_description, pad_d_emb))

			if method == 'bert':
				# print("\nmethod: {}\n".format(method))
				embedding = sess.run(
	                            tf_embedding,
	                            feed_dict={
	                                tf_description: current_description,
	                        		tf_description_length: current_length_d,
	                        		tf_bert_description_embedding: current_bert_description
	                                })

			elif method == 'lstm':
				# print("\nmethod: {}\n".format(method))
				embedding = sess.run(
	                            tf_embedding,
	                            feed_dict={
	                                tf_description: current_description,
	                        		tf_description_length: current_length_d
	                                })

			for i, ind in enumerate(current_batch_file_idx):
				desc_embed[ind] = embedding[i]
			        
		return desc_embed

	num_train = data_train['description'].shape[0]
	num_test = data_test['description'].shape[0]

	desc_embed_train = get_embeddings(num_train, data_train)
	desc_embed_test = get_embeddings(num_test, data_test)

	return num_train, desc_embed_train, num_test, desc_embed_test


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--method_', default='lstm', help='enter method bert/lstm')

	args = parser.parse_args()
	params = vars(args)
	method = params['method_']

	checkpoint_path = 'text_model_save/'+method+'/'
	
	f = h5py.File('data_text.h5', "w")

	with tf.device('/gpu:'+str(1)):	
		dataset, data_train = get_train_data(method)
		dataset, data_test = get_test_data()
		num_train, desc_embed_train, num_test, desc_embed_test = test(checkpoint_path+'model-'+str(max_itr), dataset, data_train, data_test)
		f.create_dataset('text_train', (num_train, dim_hidden), dtype='f4', data=desc_embed_train)
		f.create_dataset('text_test', (num_test, dim_hidden), dtype='f4', data=desc_embed_test)

		print("\n\nlen(desc_embed_train): {}\n\n".format(len(desc_embed_train)))
		for item in desc_embed_train[:2]:
			print(item)

		print("\n\nlen(desc_embed_test): {}\n\n".format(len(desc_embed_test)))
		for item in desc_embed_test[:2]:
			print(item)

		# with h5py.File('data_text_train.h5','r') as hf:
		# 	tem = hf.get('text_train')
		# 	print("\n\nlen(embed_train): {}\n\n".format(len(tem)))
		# 	for item in tem[:2]:
		# 		print(item)


