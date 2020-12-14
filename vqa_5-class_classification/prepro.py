"""
Preoricess a raw json dataset into hdf5/json files.

Caption: Use NLTK or split function to get tokens. 
"""
import copy
from random import shuffle, seed
import sys
import os.path
import argparse
import glob
import numpy as np
import pandas as pd
import random
# from scipy.misc import imread, imresize
# import scipy.io
import pdb
import string
import h5py
# from nltk.tokenize import word_tokenize
import json

import string
# from nltk.corpus import stopwords
from stopwords_list import stopwords
import re
def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];

def prepro_question(imgs, params):
  
    # preprocess all the question
    print ('example processed tokens:')
    for i,img in enumerate(imgs):
        s = img['question']
        # if params['token_method'] == 'nltk':
        #     txt = word_tokenize(str(s).lower())
        # else:
        txt = tokenize(str(s).lower())
        img['processed_tokens'] = txt
        # if i < 10: print (txt)
        # if i % 1000 == 0:
        #     sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(imgs), i*100.0/len(imgs)) )
        #     sys.stdout.flush()   
    return imgs

def prepro_description(imgs, params):
    # preprocess all the question
    print ('example processed description tokens:')
    for i,img in enumerate(imgs):
        s = img['description']
        # if params['token_method'] == 'nltk':
        #     txt = word_tokenize(str(s).lower())
        # else:

        # remove stopwords and punctuations
        # stop = stopwords.words('english') + list(string.punctuation)
        stop = stopwords + list(string.punctuation)
        # txt = [i for i in word_tokenize(s.lower()) if i not in stop]
        txt = [i for i in tokenize(s.lower()) if i not in stop]

        img['processed_description_tokens'] = txt
        # if i < 10: print (txt)
        # if i % 1000 == 0:
        #     sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(imgs), i*100.0/len(imgs)) )
        #     sys.stdout.flush()   
    return imgs    

def build_vocab_question(imgs, params):
    # build vocabulary for question and answers.

    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    for img in imgs:
        for w in img['processed_tokens']:
            counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
    print ('top words and their counts:')
    # print ('\n'.join(map(str,cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    print ('total words:', total_words)
    bad_words = [w for w,n in counts.items() if n <= count_thr]
    vocab = [w for w,n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print ('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
    print ('number of words in vocab would be %d' % (len(vocab), ))
    print ('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))


    # lets now produce the final annotation
    # additional special UNK token we will use below to map infrequent words to
    print ('inserting the special UNK token')
    vocab.append('UNK')
  
    for img in imgs:
        txt = img['processed_tokens']
        question = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
        img['final_question'] = question

    return imgs, vocab

def build_vocab_description(imgs, params):
    # build vocabulary for question and answers.

    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    for img in imgs:
        for w in img['processed_description_tokens']:
            counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
    print ('top words and their counts:')
    # print ('\n'.join(map(str,cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    print ('total words:', total_words)
    bad_words = [w for w,n in counts.items() if n <= count_thr]
    vocab = [w for w,n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print ('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
    print ('number of words in vocab would be %d' % (len(vocab), ))
    print ('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))


    # lets now produce the final annotation
    # additional special UNK token we will use below to map infrequent words to
    print ('inserting the special UNK token')
    vocab.append('UNK')
  
    for img in imgs:
        txt = img['processed_description_tokens']
        description = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
        img['final_description'] = description

    return imgs, vocab

def apply_vocab_question(imgs, wtoi):
    # apply the vocab on test.
    for img in imgs:
        txt = img['processed_tokens']
        question = [w if wtoi.get(w,len(wtoi)+1) != (len(wtoi)+1) else 'UNK' for w in txt]
        img['final_question'] = question

    return imgs

def apply_vocab_description(imgs, wtoi):
    # apply the vocab on test.
    for img in imgs:
        txt = img['processed_description_tokens']
        question = [w if wtoi.get(w,len(wtoi)+1) != (len(wtoi)+1) else 'UNK' for w in txt]
        img['final_description'] = question

    return imgs

def get_top_answers(imgs, params):
    counts = {}
    for img in imgs:
        ans = img['ans'] 
        counts[ans] = counts.get(ans, 0) + 1

    cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
    print ('top answer and their counts:')    
    # print ('\n'.join(map(str,cw[:20])))
    # print(len(cw))
    vocab = []
    # for top num_ans answers
    # for i in range(params['num_ans']):
    #     vocab.append(cw[i][1])

    # return vocab[:params['num_ans']]


    # for all the unique answers in train set
    for c, word in cw:
        vocab.append(word)

    print('number of unique answers: {}'.format(len(vocab)))

    return vocab

def encode_question(imgs, params, wtoi):

    max_length = params['max_length']
    N = len(imgs)

    label_arrays = np.zeros((N, max_length), dtype='uint32')
    label_length = np.zeros(N, dtype='uint32')
    question_id = np.zeros(N, dtype='uint32')
    question_counter = 0
    for i,img in enumerate(imgs):
        question_id[question_counter] = img['ques_id']
        label_length[question_counter] = min(max_length, len(img['final_question'])) # record the length of this sequence
        question_counter += 1
        for k,w in enumerate(img['final_question']):
            if k < max_length:
                label_arrays[i,k] = wtoi[w]
    
    return label_arrays, label_length, question_id

def encode_description(imgs, params, wtoi):

    max_length = 50
    N = len(imgs)

    label_arrays = np.zeros((N, max_length), dtype='uint32')
    label_length = np.zeros(N, dtype='uint32')
    # question_id = np.zeros(N, dtype='uint32')
    question_counter = 0
    for i,img in enumerate(imgs):
        # question_id[question_counter] = img['ques_id']
        label_length[question_counter] = min(max_length, len(img['final_description'])) # record the length of this sequence
        question_counter += 1
        for k,w in enumerate(img['final_description']):
            if k < max_length:
                label_arrays[i,k] = wtoi[w]
    
    return label_arrays, label_length

def encode_answer(imgs, atoi):
    N = len(imgs)
    ans_arrays = np.zeros(N, dtype='uint32')

    for i, img in enumerate(imgs):
        ans_arrays[i] = atoi[img['ans']]

    return ans_arrays

def get_multiple_choice_options(imgs, top_anss):
    options_list = []
    top_anss = pd.Series(top_anss)
    for img in imgs:
        while(True):
            options = top_anss.sample(n=4, replace=False)
            if img['ans'] not in options and not options.duplicated().any():
                break;
        options = list(options)
        options.append(img['ans'])
        random.shuffle(options)
        options_list.append(options)
    return options_list

def encode_mc_answer(imgs, options_list, atoi, type='test'):
    N = len(options_list)
    if type=='train':
        mc_ans_arrays = np.zeros((N, 5), dtype='uint32')
    else:
        mc_ans_arrays = []
    ans_indexes = []

    for i, options in enumerate(options_list):
        mc_ans_array = []
        ans_indexes.append(options.index(imgs[i]['ans']))
        for j, ans in enumerate(options):
            try:
                if type=='train':
                    mc_ans_arrays[i,j] = atoi.get(ans, 0)    
                mc_ans_array.append(str(atoi.get(ans, 0)))
            except:
                if type=='train':
                    mc_ans_arrays[i,j] = 0
                mc_ans_array.append('0')
        
        if type!='train':
            mc_ans_arrays.append(mc_ans_array)
    
    return mc_ans_arrays, ans_indexes

def get_true_ans(imgs, atoi):
    N = len(imgs)
    true_anss = np.zeros(N, dtype='uint32')

    for i, img in enumerate(imgs):
        try:
            true_anss[i] = atoi[img['ans']]
        except:
            true_anss[i] = 0
    
    return true_anss

def filter_question(imgs, atoi):
    new_imgs = []
    for i, img in enumerate(imgs):
        if atoi.get(img['ans'],len(atoi)+1) != len(atoi)+1:
            new_imgs.append(img)

    print ('question number reduce from %d to %d '%(len(imgs), len(new_imgs)))
    return new_imgs

def get_unqiue_img(imgs):
    count_img = {}
    N = len(imgs)
    img_pos = np.zeros((N, 5), dtype='uint32')
    # print(N)
    for img in imgs:
        for image in img['img_path']:
            count_img[image] = count_img.get(image, 0) + 1 

    unique_img = [w for w,n in count_img.items()]
    imgtoi = {w:i+1 for i,w in enumerate(unique_img)} # add one for torch, since torch start from 1.

    for i, img in enumerate(imgs):
        for j, image in enumerate(img['img_path']):
            img_pos[i][j] = imgtoi.get(image)

    return unique_img, img_pos

def main(params):

    imgs_train = json.load(open(params['input_train_json'], 'r'))
    imgs_test = json.load(open(params['input_test_json'], 'r'))

    # print(len(imgs_train))
    # print(len(imgs_test))

    # get top answers
    top_ans = get_top_answers(imgs_train, params)
    atoi = {w:i+1 for i,w in enumerate(top_ans)}
    itoa = {i+1:w for i,w in enumerate(top_ans)}

    # filter question, which isn't in the top answers.
    imgs_train = filter_question(imgs_train, atoi)

    seed(123) # make reproducible
    shuffle(imgs_train) # shuffle the order

    # tokenization and preprocessing training question
    imgs_train = prepro_question(imgs_train, params)
    imgs_train = prepro_description(imgs_train, params)
    # tokenization and preprocessing testing question
    imgs_test = prepro_question(imgs_test, params)
    imgs_test = prepro_description(imgs_test, params)

    # create the vocab for question
    imgs_train, vocab = build_vocab_question(imgs_train, params)
    itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
    wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table

    ques_train, ques_length_train, question_id_train = encode_question(imgs_train, params, wtoi)

    imgs_test = apply_vocab_question(imgs_test, wtoi)
    ques_test, ques_length_test, question_id_test = encode_question(imgs_test, params, wtoi)

    # create the vocab for question
    imgs_train, vocab_d = build_vocab_description(imgs_train, params)
    itow_d = {i+1:w for i,w in enumerate(vocab_d)} # a 1-indexed vocab translation table
    wtoi_d = {w:i+1 for i,w in enumerate(vocab_d)} # inverse table

    description_train, description_length_train = encode_description(imgs_train, params, wtoi_d)
    # print("\n\nlabel_arrays: {}\n\n".format(description_train))


    imgs_test = apply_vocab_description(imgs_test, wtoi_d)
    description_test, description_length_test = encode_description(imgs_test, params, wtoi_d)


    # get the unique image for train and test
    unique_img_train, img_pos_train = get_unqiue_img(imgs_train)
   
    unique_img_test, img_pos_test = get_unqiue_img(imgs_test)

    # get the answer encoding.
    A = encode_answer(imgs_train, atoi)
    if params['multiple_choice'] == 'true':
        options_list_train = get_multiple_choice_options(imgs_train, top_ans)
        MC_ans_train, ans_indexes = encode_mc_answer(imgs_train, options_list_train, atoi, 'train')
        # print("encoded ans indexes: {}".format((np.asarray(ans_indexes)+1)))
        # print("atoi: {}".format(A))

    # MC_ans_test = encode_mc_answer(imgs_test, atoi)
    test_ans = get_true_ans(imgs_test, atoi)
    test_ans_ix = {str((i+1)): str(ans) for i, ans in enumerate(test_ans)}

    # create output h5 file for training set.
    N = len(imgs_train)
    f = h5py.File(params['output_h5'], "w")
    f.create_dataset("num_answers", dtype='uint32', data=len(top_ans))
    f.create_dataset("ques_train", dtype='uint32', data=ques_train)
    f.create_dataset("ques_length_train", dtype='uint32', data=ques_length_train)
    f.create_dataset("description_train", dtype='uint32', data=description_train)
    f.create_dataset("description_length_train", dtype='uint32', data=description_length_train)
    f.create_dataset("answers", dtype='uint32', data=(np.asarray(ans_indexes)+1))
    f.create_dataset("question_id_train", dtype='uint32', data=question_id_train)
    f.create_dataset("img_pos_train", dtype='uint32', data=img_pos_train)
    if params['multiple_choice'] == 'true':
        f.create_dataset("MC_ans_train", dtype='uint32', data=MC_ans_train)
        options_list = get_multiple_choice_options(imgs_test, top_ans)
        MC_ans_test, test_ans_indexes = encode_mc_answer(imgs_test, options_list, atoi, 'train')
        # print("test_ans: {}".format(MC_ans_test))
        f.create_dataset("MC_ans_test", dtype='uint32', data=MC_ans_test)
    
    f.create_dataset("ques_test", dtype='uint32', data=ques_test)
    f.create_dataset("ques_length_test", dtype='uint32', data=ques_length_test)
    f.create_dataset("description_test", dtype='uint32', data=description_test)
    f.create_dataset("description_length_test", dtype='uint32', data=description_length_test)
    f.create_dataset("question_id_test", dtype='uint32', data=question_id_test)
    f.create_dataset("img_pos_test", dtype='uint32', data=img_pos_test)
    # f.create_dataset("MC_ans_test", dtype='uint32', data=MC_ans_test)

    f.close()
    print ('wrote ', params['output_h5'])

    # create output json file
    out = {}
    out['ix_to_word'] = itow # encode the (1-indexed) vocab
    out['ix_to_word_d'] = itow_d # encode the (1-indexed) vocab
    out['ix_to_ans'] = itoa
    out['unique_img_train'] = unique_img_train
    print("\n\n\nunique_img_train: {}\n\n\n".format(len(out['unique_img_train'])))
    out['unique_img_test'] = unique_img_test
    print("\n\n\nunique_img_test: {}\n\n\n".format(len(out['unique_img_test'])))
    
    # print(len(out['unique_img_train']))
    # print(len(out['unique_img_test']))
    # out['test_ans_ix'] = test_ans_ix
    if params['multiple_choice'] == 'true':
        # options_list = get_multiple_choice_options(imgs_test, top_ans)
        # MC_ans_test = encode_mc_answer(options_list, atoi)
        # MC_ans_test = {str((i+1)): ans for i, ans in enumerate(MC_ans_test)}
        # out['test_mc_ans_ix'] = MC_ans_test

        MC_ans_test, test_ans_indexes = encode_mc_answer(imgs_test, options_list, atoi, 'test')
        MC_ans_test = {str((i+1)): ans for i, ans in enumerate(MC_ans_test)}
        out['test_ans_ix'] = test_ans_indexes
        out['test_mc_ans_ix'] = MC_ans_test

    json.dump(out, open(params['output_json'], 'w'))
    print ('wrote ', params['output_json'])
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_train_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--input_test_json', required=True, help='input json file to process into hdf5')
    # parser.add_argument('--num_ans', type=int, help='number of top answers for the final classifications.')
    parser.add_argument('--multiple_choice', required=True, help='enter true if multiple choice')

    parser.add_argument('--output_json', default='data_prepro.json', help='output json file')
    parser.add_argument('--output_h5', default='data_prepro.h5', help='output h5 file')
  
    # options
    parser.add_argument('--max_length', default=15, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=0, type=int, help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--num_test', default=0, type=int, help='number of test images (to withold until very very end)')
    # parser.add_argument('--token_method', default='nltk', help='token method, nltk is much more slower.')

    parser.add_argument('--batch_size', default=500, type=int)

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print ('parsed input parameters:')
    print (json.dumps(params, indent = 2))

    main(params)