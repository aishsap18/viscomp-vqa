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
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def evaluate(input_stories_tensor, input_questions_tensor, input_image_features, encoder, decoder):
    '''
    Evaluates data based on the loaded checkpoint model.
    Parameters:
        input_stories_tensor (tensor): stories tensor 
        input_questions_tensor (tensor): questions tensor
        input_image_features (tensor): image features tensor
        encoder (Encoder): loaded Encoder instance
        decoder (Decoder): loaded Decoder instance
    Returns:
        results_dataset (list of tuples): tuples of format (index of input instance, generated text)
    '''
    results_dataset = []

    for i in range(0, len(input_stories_tensor), BATCH_SIZE):

        if i+BATCH_SIZE > len(input_stories_tensor):
            continue

        # record the indices of current input instances
        index = range(i, i+BATCH_SIZE)
    
        batch = i

        # if BERT variation then we already have tensors 
        # else convert the lists into tensors and select instances having indices in index 
        if 'b' not in variation:
            inp_stories = torch.tensor(input_stories_tensor)[index]
            inp_questions = torch.tensor(input_questions_tensor)[index]
        else:
            inp_stories = input_stories_tensor[index]
            inp_questions = input_questions_tensor[index]

        lens = None
        imgs = torch.tensor(input_image_features)[index]
            
        # convert the story and question tensors in the format required by encoder
        x_stories = inp_stories.transpose(0,1).to(device)
        x_questions = inp_questions.transpose(0,1).to(device)

        # initialize hidden state of encoder
        encoder_hidden = encoder.initialize_hidden_state()
        enc_output, enc_hidden, imgs_output = encoder(x_stories, x_questions, 
                                                     lens, imgs.to(device), device)
        
        # initialize decoder hidden state with encryption output hidden state
        dec_hidden = enc_hidden

        # initialize all the outputs with <start> token
        dec_input = torch.tensor([[targ_word2idx['<start>']]] * BATCH_SIZE)
        
        # initialize results array to store generated words
        results = np.zeros((max_length_tar, BATCH_SIZE))

        # run for every timestep in the batch
        for t in range(1, max_length_tar):
            predictions, dec_hidden, _ = decoder(dec_input.to(device), 
                                         dec_hidden.to(device), 
                                         enc_output.to(device),
                                         imgs_output.to(device))

            # predict the next word id
            _, predicted_ids = torch.max(predictions, dim=1)
            
            # store the predicted word id at the next position in results array 
            results[t-1] = predicted_ids.cpu()
            
            # feed the currently predicted word id as input for predicting next word 
            dec_input = predicted_ids.unsqueeze(1).long()
            

        # convert predicted ids to corresponding words and save the resulting string
        results = results.transpose()
        results_str = []
        for result in results:
            res = ''
            for r in result:
                word = targ_idx2word[r]
                if word == '<end>': 
                    break
                if word == '<start>' and word in res:
                    break
                if word != '<pad>':
                    res += targ_idx2word[r] + ' '
            
            res = res.replace('<start> ', '')
            results_str.append(res)

        temp = list(zip(index, results_str))

        results_dataset += temp

    return results_dataset


def print_save_results(data, results, num):
    '''
    Display the predictions and save results to json file.
    Parameters:
        data (array): numpy array of original data
        results (list of tuples): tuples of format (index of input instance, generated text)
        num (int): number of instances to display
    Returns:
        None
    '''

    out = []
    for (i, (ind, result)) in enumerate(results):
        
        temp = {}
        inp_story, inp_ques, targ, _ = data[ind]
        temp['input_story'] = inp_story
        temp['input_question'] = inp_ques
        temp['target'] = targ
        temp['result'] = result
        out.append(temp)

        if i < num:
            print("< {}".format(inp_story))
            print("{}".format(inp_ques))
            print("= {}".format(targ))
            print(">>> {}".format(result))
            print()

    # store the results in json file
    json.dump(out, open(results_path+'result_'+result_number+'.json', 'w'))
    print('wrote ', results_path)



def load_test_data(data):
    '''
    Function loads data from input json files and h5 datasets.
    Parameters:
        data (json): input json
    Returns:
        data (array): numpy array of original data 
        input_stories_tensor (tensor): stories tensor
        input_questions_tensor (tensor): questions tensor
        target_tensor (tensor): answer/summary tensor
        image_features (tensor): image features tensor
    '''
    global inp_lang, targ_lang, inp_stories_word2idx, inp_questions_word2idx
    
    original_input_pairs = []

    input_stories = data['input_stories_test']
    input_questions = data['input_questions_test']
    
    targets = data['targets_test']

    # load unique test image features
    with h5py.File(input_img_file, 'r') as hf:
        tem = hf.get('images_test')
        uni_image_features = np.array(tem)
    
    tem = data['img_pos_test']
    img_pos = np.array(tem)-1

    # get the image features as per position for each input instance
    image_features = []
    for i in range(len(img_pos)):
        # each input has 5 images of 4096 features
        tem = []
        for j in range(len(img_pos[i])):
            tem.append(uni_image_features[img_pos[i, j]])

        image_features.append(np.array(tem))

    image_features = np.array(image_features)
    print("image_features: {}".format(image_features.shape))

    # creating a list of tuples having the format: (story, question, answer/summary, 5 image features)
    original_input_pairs = list(zip(input_stories, input_questions, targets, image_features))

    data = np.array(original_input_pairs)

    print("data: {}".format(data.shape))
    print("---------------------------------")

    # if BERT variation then load BERT embeddings
    if 'b' in variation:
        inp_lang = None
        
        # load BERT data
        with h5py.File(input_bert_emb, 'r') as hf:
            tem = hf.get('stories_tokens_test')
            stories_tensor = np.array(tem)

            tem = hf.get('questions_tokens_test')
            questions_tensor = np.array(tem)
        
            tem = hf.get('unique_tokens_test')
            unique_tokens = np.array(tem)  

            tem = hf.get('unique_embs_test')
            unique_embs = np.array(tem)

        # create dictionary - key: tokens and value: BERT embedding
        bert_ids_to_embs = dict(zip(unique_tokens, unique_embs)) 

        # get the BERT embedding of each word in input stories and questions
        input_stories_tensor = [[bert_ids_to_embs[i] for i in inp] for inp in stories_tensor]
        input_questions_tensor = [[bert_ids_to_embs[i] for i in inp] for inp in questions_tensor]

        # convert lists to tensors
        input_stories_tensor = torch.tensor(input_stories_tensor)
        input_questions_tensor = torch.tensor(input_questions_tensor)

        print("input_stories_tensor: {}".format(input_stories_tensor.size()))
        print("input_questions_tensor: {}".format(input_questions_tensor.size()))
        print("---------------------------------------------------------------------") 

    else:
        # preprocess the stories
        data_input_stories = [preprocess_sentence(w) for w in data[:,0]]
        # get the indices of tokens in the input story based on train vocab
        input_stories_tensor = [[inp_stories_word2idx[s] for s in item.split(' ') if s in inp_stories_word2idx.keys()]  
                                    for item in data_input_stories]

        # similar to stories, pre-process and get the train vocab indices of questions
        data_input_questions = [preprocess_sentence(w) for w in data[:,1]]
        input_questions_tensor = [[inp_questions_word2idx[s] for s in item.split(' ') if s in inp_questions_word2idx.keys()]  
                                    for item in data_input_questions]
        
    # similar to stories, pre-process and get the train vocab indices of answer/summary
    data_target = [preprocess_sentence(w) for w in data[:,2]]    
    target_tensor = [[targ_word2idx[s] for s in item.split(' ') if s in targ_word2idx.keys()]  for item in data_target]
    
    print("input_stories_tensor: {}".format(np.array(input_stories_tensor).shape))
    print("input_questions_tensor: {}".format(np.array(input_questions_tensor).shape))
    print("target_tensor: {}".format(np.array(target_tensor).shape))
    print("---------------------------------")

    return data, input_stories_tensor, input_questions_tensor, target_tensor, image_features


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
    parser.add_argument('--input_bert_emb', default=None, help='enter input bert embeddings file')
    parser.add_argument('--checkpoint_path', required=True, help='enter checkpoint path')
    parser.add_argument('--results_path', required=True, help='enter results path')

    args = parser.parse_args()
    params = vars(args)

    variation = params['variation']
    input_data_file = params['input_data_file']
    input_img_file = params['input_img_file']
    input_bert_emb = params['input_bert_emb']
    checkpoint_path = params['checkpoint_path']
    results_path = params['results_path']
    result_number = checkpoint_path.split('-')[1].replace('.pt', '')

    # load the checkpoint model
    checkpoint = torch.load(checkpoint_path)

    # set the parameters based on the loaded model
    BATCH_SIZE = checkpoint["BATCH_SIZE"]
    embedding_dim = checkpoint["embedding_dim"]
    img_dim = checkpoint["img_dim"]
    units = checkpoint["units"]
    vocab_tar_size = checkpoint["vocab_tar_size"]

    vocab_inp_stories_size = checkpoint["vocab_inp_stories_size"]
    vocab_inp_questions_size = checkpoint["vocab_inp_questions_size"]
    inp_questions_word2idx = checkpoint["inp_questions_word2idx"]
    inp_stories_word2idx = checkpoint["inp_stories_word2idx"]

    targ_word2idx = checkpoint["targ_word2idx"]
    targ_idx2word = checkpoint["targ_idx2word"]
    max_length_tar = checkpoint["max_length_tar"]

    if 'g' in variation:
        stories_weights_matrix = checkpoint["stories_weights_matrix"]
        questions_weights_matrix = checkpoint["questions_weights_matrix"]
        target_weights_matrix = checkpoint["target_weights_matrix"]
    else:
        stories_weights_matrix = None
        questions_weights_matrix = None
        target_weights_matrix = None

    # initilize encoder with the parameters above and load the corresponding weights
    encoder = Encoder(vocab_inp_stories_size, vocab_inp_questions_size, embedding_dim, units, 
                      BATCH_SIZE, img_dim, variation, stories_weights_matrix, questions_weights_matrix)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.to(device)

    # initilize decoder with the parameters above and load the corresponding weights
    decoder = Decoder(vocab_tar_size, embedding_dim, units, units, BATCH_SIZE, variation, target_weights_matrix)
    decoder.load_state_dict(checkpoint["decoder"])
    decoder.to(device)

    # evaluate
    encoder.eval()
    decoder.eval()

    # load input val/test data
    data_test = json.load(open(input_data_file, 'r'))
    data, input_stories_tensor, input_questions_tensor, target_tensor, input_image_features = load_test_data(data_test)

    # if any other variation than BERT then pad the stories and questions to maximum length
    # in case of BERT this is already done in prepro_data.py
    if 'b' not in variation:
        max_length_inp_stories, max_length_inp_questions = max_length(input_stories_tensor), max_length(input_questions_tensor)
        input_stories_tensor = [pad_sequences(x, max_length_inp_stories) for x in input_stories_tensor]
        input_questions_tensor = [pad_sequences(x, max_length_inp_questions) for x in input_questions_tensor]
        print("len(input_stories_tensor): {}".format(len(input_stories_tensor)))
        print("len(input_questions_tensor): {}".format(len(input_questions_tensor)))
        print("---------------------------------")

    # evaluate data based on the loaded encoder-decoder and generate answer/summary
    results = evaluate(input_stories_tensor, input_questions_tensor, input_image_features, encoder, decoder)

    # display and store results
    print_save_results(data, results, 5)

