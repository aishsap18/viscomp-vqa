import math
import json
import argparse
import numpy as np
import h5py
from sentence_transformers import SentenceTransformer


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--input_train_json', required=True, help='enter input train json file')
	parser.add_argument('--input_test_json', required=True, help='enter input test json file')
	parser.add_argument('--out_name', default='data_text_bert.h5', help='enter output file name')

	args = parser.parse_args()
	params = vars(args)
	input_train_json = params['input_train_json']
	input_test_json = params['input_test_json']
	out_name = params['out_name']

	with open(input_train_json) as f:
		data_train = json.load(f)

	with open(input_test_json) as f:
		data_test = json.load(f)

	sentences_train = [d['description'] for d in data_train]
	print("Number of train descriptions: {}".format(len(sentences_train)))

	sentences_test = [d['description'] for d in data_test]
	print("Number of test descriptions: {}".format(len(sentences_test)))	

	model = SentenceTransformer('bert-base-nli-mean-tokens')
	
	sentence_embeddings_train = np.array(model.encode(sentences_train))
	sentence_embeddings_test = np.array(model.encode(sentences_test))
	# print(sentence_embeddings_train)
	# print(sentence_embeddings_test)

	f = h5py.File(out_name, "w")
	f.create_dataset('text_embeddings_train', sentence_embeddings_train.shape, dtype='f4', data=sentence_embeddings_train)
	f.create_dataset('text_embeddings_test', sentence_embeddings_test.shape, dtype='f4', data=sentence_embeddings_test)
