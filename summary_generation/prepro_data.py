import sys
import os.path
import argparse
import numpy as np
import h5py
import json
import re
import unicodedata

import torch

from transformers import BertModel, BertTokenizer


def get_unqiue_img(imgs):
    '''
    Returns a list of unique image ids along with their corresponding positions in the new list.
    Parameters:
        imgs (list of objects): raw input data
    Returns:
        unique_img (list of str): list of unique image ids
        img_pos (list of list): list of indices for each input instance corresponding to the unique_img list
    '''

    count_img = {}
    N = len(imgs)
    # each input instance has 5 images, so initializing a 2D array of size (total # of instances, 5)
    img_pos = np.zeros((N, 5), dtype='uint32')
   
    # creating dictionary with key: image ids and value: count
    for img in imgs:
        for image in img['img_path']:
            count_img[image] = count_img.get(image, 0) + 1 

    # list of unique image ids
    unique_img = [w for w,n in count_img.items()]
    # adding one for torch, since torch start from 1
    imgtoi = {w:i+1 for i,w in enumerate(unique_img)} 

    # assigning the corresponding index from unique_img to the input instances
    for i, img in enumerate(imgs):
        for j, image in enumerate(img['img_path']):
            img_pos[i][j] = imgtoi.get(image)

    return unique_img, img_pos


def max_length(tensor):
    '''
    Returns the length of maximum tensor.
    Parameters:
        tensor (tensor of tensors): list of tensors
    Returns:
        (int): max length
    '''
    return max(len(t) for t in tensor)


def pad_sequences(x, max_len):
    '''
    Pads a tensor with the given length.
    Parameters:
        x (tensor): original tensor
        max_len (int): length to pad the tensor with
    Returns:
        padded (tensor): padded tensor
    '''

    padded = np.zeros((max_len), dtype=np.int64)
    if len(x) > max_len: padded[:] = x[:max_len]
    else: padded[:len(x)] = x
    return padded


# loading BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_tokens(text):
    '''
    Tokenize the text.
    Parameters: 
        text (str): input text
    Returns: 
        (list of tokens): tokenized text
    '''
    return tokenizer.encode(text, add_special_tokens=True)


def get_bert_embeddings(tokens):
    '''
    Extracts BERT embeddings based on tokens.     
    Parameters:
        tokens (list of tokens): tokenized text
    Returns:
        word_emb (tensor): BERT word embeddings
    '''
    word_emb = model(tokens)[0][0]
    return word_emb


def unicode_to_ascii(s):
    '''
    Converts the unicode to ascii. Normalizes latin chars with accent to their canonical decomposition.
    Parameters:
        s (str): input string
    Returns:
        (str): ascii converted string
    '''
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def clean_text(w):
    '''
    Preprocesses the input text.
    Parameters:
        w (str): raw string
    Returns:
        w (str): cleaned string
    '''

    # convert to lower case string, strip extra spaces and convert to ascii
    w = unicode_to_ascii(w.lower().strip())
    # adds a space between word and punctuations (.!?)
    w = re.sub(r"([.!?])", r" \1", w)
    # allows only numbers, upper & lower case letters and '.!?' punctuations
    w = re.sub(r"[^\da-zA-Z.!?]+", r" ", w)
    return w


def load_data(imgs):
    '''
    Loads the data in required format.
    Parameters:
        imgs (list of objects): raw input data
    Returns:
        input_stories (tensor): stories tensor 
        input_questions (tensor): questions tensor
        stories_tokens (tensor): tokenized stories
        questions_tokens (tensor): tokenized questions
        targets (tensor): target (summary or answer) tensor 
        img_pos (tensor): 2D tensor with image indices
        unique_img (tensor): tensor of unique images
        u_t (tensor): unique tokens list for BERT
        u_e (tensor): unique words' BERT embeddings 
    '''

    input_stories = []
    input_questions = []
    targets = []

    stories_tokens = []
    questions_tokens = []

    for img in imgs:
        input_stories.append(img['description'])
        input_questions.append(img['question'])

        # if BERT variation
        if 'b' in variation:
            # clean the text and get tokens
            cleaned_text = clean_text(img['description'])
            stories_tokens.append(get_tokens(cleaned_text))
            cleaned_text = clean_text(img['question'])
            questions_tokens.append(get_tokens(cleaned_text))

        target_s = img['ans']
        targets.append(target_s)

    # getting the unique images
    unique_img, img_pos = get_unqiue_img(imgs)
    u_t, u_e = None, None

    # if BERT variation 
    if 'b' in variation:
        # get the max length story and question
        max_length_inp_stories, max_length_inp_questions = max_length(stories_tokens), max_length(questions_tokens) 
        
        # pad all the input stories and questions with max length
        stories_tokens = [pad_sequences(x, max_length_inp_stories) for x in stories_tokens]
        questions_tokens = [pad_sequences(x, max_length_inp_questions) for x in questions_tokens]

        # getting the unique story and question tokens
        unique_tokens_stories = set(x for l in stories_tokens for x in l)
        unique_tokens_questions = set(x for l in questions_tokens for x in l)
        # combining both of them together
        unique_tokens_list = np.array(list(unique_tokens_stories.union(unique_tokens_questions)))
        print("unique_tokens_list: {}".format(unique_tokens_list.shape))

        # splitting them into chunks/batches of 512 size 
        chunk_size = (unique_tokens_list.shape[0] / 512) + 1
        chunks = np.array_split(unique_tokens_list, chunk_size)
        
        # getting the BERT embeddings
        unique_embs = []
        for chunk in chunks:
            unique_embs.append(get_bert_embeddings(torch.tensor([chunk])))

        u_e = torch.cat(unique_embs, 0).detach().numpy()
        u_t = unique_tokens_list

    return input_stories, input_questions, np.array(stories_tokens), np.array(questions_tokens), targets, img_pos, unique_img, u_t, u_e


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--train_input', required=True, help='enter train input json')
    parser.add_argument('--test_input', required=True, help='enter test input json')
    parser.add_argument('--output_json', required=True, help='enter output json')
    parser.add_argument('--output_bert_h5', required=True, help='enter output bert h5')
    parser.add_argument('--variation', required=True, help='enter variation - isq, iq, sq, bsq, bisq, biq')

    args = parser.parse_args()
    params = vars(args)

    variation = params['variation']

    imgs_train = json.load(open(params['train_input'], 'r'))
    imgs_test = json.load(open(params['test_input'], 'r'))
    
    out = {}

    # load the train and val data
    input_stories, input_questions, stories_tokens, questions_tokens, targets, img_pos, unique_img, unique_tokens, unique_embs = load_data(imgs_train)
    input_stories_test, input_questions_test, stories_tokens_test, questions_tokens_test, targets_test, img_pos_test, unique_img_test, unique_tokens_test, unique_embs_test = load_data(imgs_test)
    
    out['img_pos'] = img_pos.tolist()
    out['unique_img_train'] = unique_img
    print("len(unique_img): {}".format(len(unique_img)))        
 
    out['input_stories'] = input_stories
    out['input_questions'] = input_questions
    out['targets'] = targets

    out['img_pos_test'] = img_pos_test.tolist()
    out['unique_img_test'] = unique_img_test

    out['input_stories_test'] = input_stories_test
    out['input_questions_test'] = input_questions_test
    out['targets_test'] = targets_test

    # if BERT variation then store the BERT embeddings into h5 dataset
    if 'b' in variation:
        f = h5py.File(params['output_bert_h5'], "w")
       
        f.create_dataset('stories_tokens', stories_tokens.shape, dtype='int64', data=stories_tokens)
        f.create_dataset('questions_tokens', questions_tokens.shape, dtype='int64', data=questions_tokens)

        f.create_dataset('unique_tokens', unique_tokens.shape, dtype='int64', data=unique_tokens)
        f.create_dataset('unique_embs', unique_embs.shape, dtype='f4', data=unique_embs)

        f.create_dataset('stories_tokens_test', stories_tokens_test.shape, dtype='int64', data=stories_tokens_test)
        f.create_dataset('questions_tokens_test', questions_tokens_test.shape, dtype='int64', data=questions_tokens_test)

        f.create_dataset('unique_tokens_test', unique_tokens_test.shape, dtype='int64', data=unique_tokens_test)
        f.create_dataset('unique_embs_test', unique_embs_test.shape, dtype='f4', data=unique_embs_test)

    # store the formatted data in json file
    json.dump(out, open(params['output_json'], 'w'))
    print('wrote ', params['output_json'])
