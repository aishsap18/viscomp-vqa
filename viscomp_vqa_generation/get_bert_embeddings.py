import math
import json
import argparse
import numpy as np
import h5py
from sentence_transformers import SentenceTransformer


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	# parser.add_argument('--input_file', required=True, help='enter input json file')
	parser.add_argument('--variation', required=True, help='q or sq')
	parser.add_argument('--train_input', required=True, help='enter train input json')
    parser.add_argument('--test_input', required=True, help='enter test input json')

	args = parser.parse_args()
	params = vars(args)
	variation = params['variation']


	# input_file = params['input_file']

	with open(params['train_input']) as f:
		data_train = json.load(f)

	with open(params['test_input']) as f:
		data_test = json.load(f)

	if 's' in variation:
		sentences_train = [d['description'] + ' ' + d['question'] for d in data_train]
	else:
		sentences_train = [d['question'] for d in data_train]
	print("Number of train instances: {}".format(len(sentences_train)))

	if 's' in variation:
		sentences_test = [d['description'] + ' ' + d['question'] for d in data_test]
	else:
		sentences_test = [d['question'] for d in data_test]
	print("Number of test descriptions: {}".format(len(sentences_test)))	

	model = SentenceTransformer('bert-base-nli-mean-tokens')
	
	sentence_embeddings_train = np.array(model.encode(sentences_train))
	sentence_embeddings_test = np.array(model.encode(sentences_test))
	print(sentence_embeddings_train.shape)
	# print(sentence_embeddings_train[0:3])
	print(sentence_embeddings_test)

	f = h5py.File('data_text_bert_'+variation+'_'+str(len(sentences_train))+'.h5', "w")
	f.create_dataset('text_embeddings_train', sentence_embeddings_train.shape, dtype='f4', data=sentence_embeddings_train)
	f.create_dataset('text_embeddings_test', sentence_embeddings_test.shape, dtype='f4', data=sentence_embeddings_test)
