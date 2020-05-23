import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

from batch_model import *
from batch_train_pytorch import *

import numpy as np
import unicodedata
import re
import time

import json
import h5py
import argparse

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# evaluate

def evaluate(dataset, encoder, decoder):
    results_dataset = []

    for (batch, (inp, targ, inp_len, imgs)) in enumerate(dataset):
    # for i in range(0, 1):
    #     batch = 0
    #     inp, targ, inp_len, imgs = next(iter(dataset))

        if 'b' in variation:
            xs = inp.transpose(0,1)
            lens = None
            idx = np.array(range(batch*BATCH_SIZE, batch*BATCH_SIZE+BATCH_SIZE))
        else:
            xs, _, lens, imgs, idx = sort_batch(inp, None, inp_len, imgs, True)

        encoder_hidden = encoder.initialize_hidden_state(device)
        enc_output, enc_hidden = encoder(xs.to(device), lens, imgs.to(device), device)
        dec_hidden = enc_hidden

        dec_input = torch.tensor([[targ_word2idx['<start>']]] * BATCH_SIZE)
        
        results = np.zeros((max_length_tar, BATCH_SIZE))

        for t in range(1, max_length_tar):
            predictions, dec_hidden, _ = decoder(dec_input.to(device), 
                                         dec_hidden.to(device), 
                                         enc_output.to(device))
            
            dec_input, predicted_ids = torch.max(predictions, dim=1)
            
            results[t-1] = predicted_ids.cpu()
            
            dec_input = dec_input.unsqueeze(1).long()


        results = results.transpose()

        results_str = []
        for result in results:
            res = ''
            for r in result:
                if targ_idx2word[r] == '<end>': 
                    break
                if targ_idx2word[r] != '<pad>':
                    res += targ_idx2word[r] + ' '
            results_str.append(res)

        if 'b' in variation:
            temp = list(zip(idx, results_str))
        else:
            temp = list(zip(idx.numpy(), results_str))

        results_dataset += temp

    return results_dataset


def print_save_results(data, results, num):

    out = []
    for (i, (ind, result)) in enumerate(results):
        
        temp = {}
        inp, targ, _ = data[ind]
        temp['input'] = inp
        temp['target'] = targ
        temp['result'] = result
        out.append(temp)

        if i < num:
            print("< {}".format(inp))
            print("= {}".format(targ))
            print(">>>>>>> {}".format(result))
            print()

    json.dump(out, open(results_path+variation+'/result_'+result_number+'.json', 'w'))
    print('wrote ', results_path)



def load_test_data(data):
    global inp_lang, targ_lang
    
    original_input_pairs = []

    inputs = data['inputs_test']
    targets = data['targets_test']

    with h5py.File(input_img_file, 'r') as hf:
        tem = hf.get('images_test')
        uni_image_features = np.array(tem)
    
    tem = data['img_pos_test']
    img_pos = np.array(tem)-1

    image_features = []
    for i in range(len(img_pos)):
        # each input has 5 images of 4096 features
        tem = []
        for j in range(len(img_pos[i])):
            tem.append(uni_image_features[img_pos[i, j]])

        image_features.append(np.array(tem))

    image_features = np.array(image_features)
    print("image_features: {}".format(image_features.shape))

    original_input_pairs = list(zip(inputs, targets, image_features))

    data = np.array(original_input_pairs)

    print("data: {}".format(data.shape))
    print("---------------------------------")

    if 'b' in variation:
        inp_lang = None
        with h5py.File(input_bert_emb, 'r') as hf:
            tem = hf.get('text_embeddings_test')
            input_tensor = np.array(tem)
    else:
        data_input = [preprocess_sentence(w) for w in data[:,0]]
        input_tensor = [[inp_word2idx[s] for s in item.split(' ') if s in inp_word2idx.keys()] for item in data_input]

    data_target = [preprocess_sentence(w) for w in data[:,1]]
    
    # Vectorize the input and target languages
    target_tensor = [[targ_word2idx[s] for s in item.split(' ') if s in targ_word2idx.keys()]  for item in data_target]
    print("input_tensor: {}".format(np.array(input_tensor).shape))
    print("target_tensor: {}".format(np.array(target_tensor).shape))
    print("---------------------------------")
    
    return data, input_tensor, target_tensor, image_features


BATCH_SIZE = 0
embedding_dim = 0
img_dim = 0
units = 0
vocab_inp_size = 0 
vocab_tar_size = 0 
targ_lang = None 

variation = ''

input_data_file = ''
input_img_file = ''
input_bert_emb = ''
checkpoint_path = ''
results_path = ''
result_number = ''

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--variation', required=True, help='enter variation - isq, iq, sq, q, bsq, bq, bisq, biq')
    parser.add_argument('--input_data_file', required=True, help='enter input data file')
    parser.add_argument('--input_img_file', required=True, help='enter input image file')
    parser.add_argument('--input_bert_emb', required=True, help='enter input bert embeddings file')
    parser.add_argument('--checkpoint_path', required=True, help='enter checkpoint path')
    parser.add_argument('--results_path', required=True, help='enter results path')

    args = parser.parse_args()
    params = vars(args)

    # variation = isq, iq, sq, q, bsq, bq, bisq, biq (b - bert embeddings)
    variation = params['variation']
    input_data_file = params['input_data_file']
    input_img_file = params['input_img_file']
    input_bert_emb = params['input_bert_emb']
    checkpoint_path = params['checkpoint_path']
    results_path = params['results_path']
    result_number = checkpoint_path.split('-')[1].replace('.pt', '')

    checkpoint = torch.load(checkpoint_path)

    BATCH_SIZE = checkpoint["BATCH_SIZE"]
    embedding_dim = checkpoint["embedding_dim"]
    img_dim = checkpoint["img_dim"]
    units = checkpoint["units"]
    vocab_inp_size = checkpoint["vocab_inp_size"]
    vocab_tar_size = checkpoint["vocab_tar_size"]

    inp_word2idx = checkpoint["inp_word2idx"]
    targ_word2idx = checkpoint["targ_word2idx"]
    targ_idx2word = checkpoint["targ_idx2word"]
    max_length_tar = checkpoint["max_length_tar"]

    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE, img_dim, variation)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.to(device)

    decoder = Decoder(vocab_tar_size, embedding_dim, units, units, BATCH_SIZE, variation)
    decoder.load_state_dict(checkpoint["decoder"])
    decoder.to(device)

    encoder.eval()
    decoder.eval()

    data_test = json.load(open(input_data_file, 'r'))
    data, input_tensor, target_tensor, input_image_features = load_test_data(data_test)

    max_length_inp = max_length(input_tensor)

    # inplace padding
    input_tensor_test = [pad_sequences(x, max_length_inp) for x in input_tensor]
    target_tensor_test = [pad_sequences(x, max_length_tar) for x in target_tensor]

    test_dataset = PrepareDataForDataLoader(input_tensor_test, target_tensor_test, input_image_features)

    test_dataset = DataLoader(test_dataset, batch_size = BATCH_SIZE, 
                     drop_last=True,
                     shuffle=False)

    results = evaluate(test_dataset, encoder, decoder)

    print_save_results(data, results, 5)
