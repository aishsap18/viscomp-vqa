import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

from batch_model import *

import numpy as np
import unicodedata
import re
import time

import json
import h5py
import argparse

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    """
    Normalizes latin chars with accent to their canonical decomposition
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ." 
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    # w = re.sub(r"([?.!,¿])", r" \1 ", w)
    # w = re.sub(r'[" "]+', " ", w)
    
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    # w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    
    w = re.sub(r"([.!?])", r" \1", w)
    w = re.sub(r"[^\da-zA-Z.!?]+", r" ", w)

    # w = w.rstrip().strip()
    
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


# This class creates a word -> index mapping (e.g,. "dad" -> 5) and vice-versa 
# (e.g., 5 -> "dad") for each language,
class LanguageIndex():
    def __init__(self, lang, test=False):
        """ lang are the list of phrases from each language"""
        if test == False:
            self.lang = lang
            self.word2idx = {}
            self.idx2word = {}
            self.vocab = set()
            
            self.create_index()

        
    def create_index(self):
        for phrase in self.lang:
            # update with individual tokens
            self.vocab.update(phrase.split(' '))
            
        # sort the vocab
        self.vocab = sorted(self.vocab)

        # add a padding token with index 0
        self.word2idx['<pad>'] = 0
        
        # word to index mapping
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1 # +1 because of pad token
        
        # index to word mapping
        for word, index in self.word2idx.items():
            self.idx2word[index] = word


def max_length(tensor):
    return max(len(t) for t in tensor)


def pad_sequences(x, max_len):
    padded = np.zeros((max_len), dtype=np.int64)
    if len(x) > max_len: padded[:] = x[:max_len]
    else: padded[:len(x)] = x
    return padded


# convert the data to tensors and pass to the Dataloader 
# to create an batch iterator
class PrepareDataForDataLoader(Dataset):
    def __init__(self, X, y, input_image_features):
        self.data = X
        self.target = y
        self.images = input_image_features
        self.length = [ np.sum(1 - np.equal(x, 0)) for x in X]
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        x_len = self.length[index]
        i = self.images[index]
        return x,y,x_len, i
    
    def __len__(self):
        return len(self.data)

### sort batch function to be able to use with pad_packed_sequence
def sort_batch(X, y, lengths, imgs, test=False):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    if test == False:
        y = y[indx]
    imgs = imgs[indx]
    return X.transpose(0,1), y, lengths, imgs, indx # transpose (batch x seq) to (seq x batch)

criterion = nn.CrossEntropyLoss()

def loss_function(real, pred):
    """ Only consider non-zero inputs in the loss; mask needed """
    #mask = 1 - np.equal(real, 0) # assign 0 to all above 0 and 1 to all 0s
    #print(mask)
    mask = real.ge(1).type(torch.cuda.FloatTensor)
    
    loss_ = criterion(pred, real) * mask 
    return torch.mean(loss_)

def train(dataset):

    ## TODO: Combine the encoder and decoder into one class
    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE, img_dim, variation)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, units, BATCH_SIZE, variation)

    encoder.to(device)
    decoder.to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), 
                           lr=0.001)

    EPOCHS = 100
    itr = 0

    for epoch in range(EPOCHS+1):
        start = time.time()
        
        encoder.train()
        decoder.train()
        
        total_loss = 0
        # i = 0
        for (batch, (inp, targ, inp_len, imgs)) in enumerate(dataset):
        # for i in range(0, 1):
        #     batch = 0
        #     inp, targ, inp_len, imgs = next(iter(dataset))
            # print("imgs in train: {}".format(imgs.shape))
            itr = itr + 1
            loss = 0
            
            # _ indexes
            xs, ys, lens, imgs, _ = sort_batch(inp, targ, inp_len, imgs)
                       
            enc_output, enc_hidden = encoder(xs.to(device), lens, imgs.to(device), device)
            dec_hidden = enc_hidden
            
            # use teacher forcing - feeding the target as the next input (via dec_input)
            dec_input = torch.tensor([[targ_lang.word2idx['<start>']]] * BATCH_SIZE)
            
            # run code below for every timestep in the ys batch
            for t in range(1, ys.size(1)):
                predictions, dec_hidden, _ = decoder(dec_input.to(device), 
                                             dec_hidden.to(device), 
                                             enc_output.to(device))
                loss += loss_function(ys[:, t].to(device), predictions.to(device))
                #loss += loss_
                dec_input = ys[:, t].unsqueeze(1)
                
            
            batch_loss = (loss / int(ys.size(1)))
            total_loss += batch_loss
            
            optimizer.zero_grad()
            
            loss.backward()

            ### UPDATE MODEL PARAMETERS
            optimizer.step()
            
            if batch % 100 == 0:
                print('Epoch {} Iteration {} Batch {} Loss {:.4f}'.format(epoch,
                                                             itr,
                                                             batch,
                                                             batch_loss.detach().item()))
        
        if epoch % 20 == 0:    
            if inp_lang != None:
                inp_word2idx = inp_lang.word2idx
            else:
                inp_word2idx = None

            torch.save({
                        "BATCH_SIZE": BATCH_SIZE,
                        "vocab_inp_size": vocab_inp_size,
                        "vocab_tar_size": vocab_tar_size, 
                        "embedding_dim": embedding_dim,
                        "img_dim": img_dim, 
                        "units": units,
                        "max_length_tar": max_length_tar,
                        "encoder": encoder.state_dict(), 
                        "decoder": decoder.state_dict(),
                        "inp_word2idx": inp_word2idx,
                        "targ_word2idx": targ_lang.word2idx,
                        "targ_idx2word": targ_lang.idx2word

                }, checkpoint_path+"model_"+variation+"/epoch-"+str(epoch)+".pt")
        
        print('Epoch {} Iteration {} Loss {:.4f}'.format(epoch, 
                                            itr,
                                            total_loss / N_BATCH))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    print("---------------------------------")


def load_data(data):
    global inp_lang, targ_lang

    original_input_pairs = []

    inputs = data['inputs']
    targets = data['targets']

    with h5py.File(input_img_file, 'r') as hf:
        tem = hf.get('images_train')
        uni_image_features = np.array(tem)
    
    tem = data['img_pos']
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
            tem = hf.get('text_embeddings_train')
            input_tensor = np.array(tem)
    else:
        data_input = [preprocess_sentence(w) for w in data[:,0]]
        inp_lang = LanguageIndex(data_input)
        input_tensor = [[inp_lang.word2idx[s] for s in item.split(' ')]  for item in data_input]

    data_target = [preprocess_sentence(w) for w in data[:,1]]

    # index language using the class above
    targ_lang = LanguageIndex(data_target)
    
    # Vectorize the input and target languages
    target_tensor = [[targ_lang.word2idx[s] for s in item.split(' ')]  for item in data_target]
    print("input_tensor: {}".format(np.array(input_tensor).shape))
    print("target_tensor: {}".format(np.array(target_tensor).shape))
    print("---------------------------------")

    return data, input_tensor, target_tensor, image_features


BUFFER_SIZE = 0 # len(input_tensor_train)
BATCH_SIZE = 0
N_BATCH = 0 # BUFFER_SIZE//BATCH_SIZE
embedding_dim = 0
units = 0
img_dim = 0
vocab_inp_size = 0 # len(inp_lang.word2idx)
vocab_tar_size = 0 # len(targ_lang.word2idx)
inp_lang = None # LanguageIndex(data_input)
targ_lang = None # LanguageIndex(data_target)

variation = ''

input_data_file = ''
input_img_file = ''
input_bert_emb = ''

checkpoint_path = ''

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--variation', required=True, help='enter variation - isq, iq, sq, q, bsq, bq, bisq, biq')
    parser.add_argument('--input_data_file', required=True, help='enter input data file')
    parser.add_argument('--input_img_file', required=True, help='enter input image file')
    parser.add_argument('--input_bert_emb', required=True, help='enter input bert embeddings file')
    parser.add_argument('--model_save', required=True, help='enter model save path')

    args = parser.parse_args()
    params = vars(args)

    # variation = isq, iq, sq, q, bsq, bq, bisq, biq (b - bert embeddings)
    variation = params['variation']
    input_data_file = params['input_data_file']
    input_img_file = params['input_img_file']
    input_bert_emb = params['input_bert_emb']
    checkpoint_path = params['model_save']

    data_train = json.load(open(input_data_file, 'r'))

    _, input_tensor, target_tensor, input_image_features = load_data(data_train)

    # calculate the max_length of input and output tensor
    max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)
    print("max_length_inp: {}".format(max_length_inp))
    print("max_length_tar: {}".format(max_length_tar))

    # inplace padding
    input_tensor_train = [pad_sequences(x, max_length_inp) for x in input_tensor]
    target_tensor_train = [pad_sequences(x, max_length_tar) for x in target_tensor]
    print("len(input_tensor_train): {}".format(len(input_tensor_train)))
    print("len(target_tensor_train): {}".format(len(target_tensor_train)))
    print("---------------------------------")


    BUFFER_SIZE = len(input_tensor_train)
    BATCH_SIZE = 16 # 64
    N_BATCH = BUFFER_SIZE//BATCH_SIZE
    embedding_dim = 256
    img_dim = 4096
    units = 1024
    
    if 'b' in variation:
        vocab_inp_size = len(input_tensor_train[0])
    else:
        vocab_inp_size = len(inp_lang.word2idx)

    vocab_tar_size = len(targ_lang.word2idx)
    
    print("vocab_inp_size: {}".format(vocab_inp_size))
    print("vocab_tar_size: {}".format(vocab_tar_size))
    print("---------------------------------")

    train_dataset = PrepareDataForDataLoader(input_tensor_train, target_tensor_train, input_image_features)

    dataset = DataLoader(train_dataset, batch_size = BATCH_SIZE, 
                         drop_last=True,
                         shuffle=True)

    train(dataset)
