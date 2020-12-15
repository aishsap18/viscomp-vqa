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

import torchtext

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def preprocess_sentence(w):
    '''
    Preprocesses the input text and adds <start> and <end> token.
    Parameters:
        w (str): raw string
    Returns:
        w (str): processed string
    '''

    # convert to lower case string, strip extra spaces and convert to ascii
    w = unicode_to_ascii(w.lower().strip())
    # adds a space between word and punctuations (.!?)
    w = re.sub(r"([.!?])", r" \1", w)
    # allows only numbers, upper & lower case letters and '.!?' punctuations
    w = re.sub(r"[^\da-zA-Z.!?]+", r" ", w)

    # adding a start and end token to the sentence
    if 'b' not in variation or 'g' not in variation:
        w = '<start> ' + w + ' <end>'
    else:
        w = '<start> ' + w + ' <end>'
    return w


class LanguageIndex():
    '''
    Creates a word to index and index to word mapping
    '''
    def __init__(self, lang, test=False):
        if test == False:
            # sentences
            self.lang = lang
            self.word2idx = {}
            self.idx2word = {}
            # vocabulary
            self.vocab = set()

            # if GloVe variation
            if 'g' in variation:
                self.idx2gemb = {}
            
            # populate the above variables
            self.create_index()

        
    def create_index(self):
        '''
        Populating all the class variables by creating indices for each word in the vocab.
        Parameters:
            None
        Returns:
            None
        '''

        # get the words/tokens from each sentences and create a flat list
        for phrase in self.lang:
            # update with individual tokens
            self.vocab.update(phrase.split(' '))
            
        # sort the vocab tokens
        self.vocab = sorted(self.vocab)

        # add pad token with index 0
        self.word2idx['<pad>'] = 0
        
        # map word to index
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1 
        
        # map index to word
        for word, index in self.word2idx.items():
            self.idx2word[index] = word
            
            # if GloVe variation then extract and store GloVe embedding and map index to embedding
            if 'g' in variation:
                emb = glove_embs[word]
                if emb[0] == 0:
                    self.idx2gemb[index] = torch.tensor(np.random.uniform(low=-1, high=1, size=(50,)), dtype=torch.float32)
                else:
                    self.idx2gemb[index] = emb


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


class PrepareDataForDataLoader(Dataset):
    '''
    Passing the data to Dataloader to create batch iterator.
    '''
    def __init__(self, stories, questions, answers, input_image_features):
        self.stories = stories
        self.questions = questions
        self.target = answers
        self.images = input_image_features
        
        # if BERT is not to be used then store the lengths of the stories and questions
        if 'b' not in variation:
            self.length_s = [ np.sum(1 - np.equal(x, 0)) for x in stories]
            self.length_q = [ np.sum(1 - np.equal(x, 0)) for x in questions]
        
    def __getitem__(self, index):
        stories = self.stories[index]
        questions = self.questions[index]
        answers = self.target[index]
        images = self.images[index]

        # if BERT is not to be used then store the lengths of the stories and questions
        if 'b' not in variation:
            stories_len = self.length_s[index]
            questions_len = self.length_q[index]
            return stories,questions, answers,stories_len, questions_len, images
        else:
            return stories,questions, answers, images
    
    def __len__(self):
        return len(self.questions)


def sort_batch(X, y, lengths, imgs, test=False, indexes=None):
    '''
    Sort the batch instances by length so that pad_packed_sequence can be used.
    Parameters:
        X (tensor): input tensor
        y (tensor): target tensor
        lengths (tensor): length of each input
        imgs (tensor): images tensor
        test (bool): train/test mode 
        indexes (tensor): tensor of indices of the input
    Returns:
        X (tensor): sorted input
        y (tensor): sorted target
        lengths (tensor): sorted lengths
        imgs (tensor): sorted imgs
        indexes (tensor): sorted indices
    '''
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    # if in test mode then do not sort target tensor
    if test == False:
        y = y[indx]
    else:
        indexes = indexes[indx]
    imgs = imgs[indx]
    return X.transpose(0,1), y, lengths, imgs, indexes 


# Cross Entropy Loss
criterion = nn.CrossEntropyLoss()

def loss_function(real, pred):
    '''
    Calculating loss.
    Parameters:
        real (tensor): ground truth word
        pred (tensor): predicted word
    Returns:
        (float): loss
    '''

    # considering only non-zero inputs in the loss
    mask = real.ge(1).type(torch.cuda.FloatTensor)
    loss_ = criterion(pred, real) * mask 
    return torch.mean(loss_)


def get_emb_weight_matrix(lang):
    '''
    Returns unsqueezed GloVe embeddings to match the format for loading using torch.nn.Embedding.
    Parameters: 
        lang (LanguageIndex): LanguageIndex class instance
    Returns:
        weights_matrix (tensor): unsqueezed tensor of embeddings
    '''
    weights_matrix = list(lang.idx2gemb.values())
    weights_matrix = [torch.unsqueeze(t, 0) for t in weights_matrix]
    weights_matrix = torch.cat(weights_matrix, 0)
    return weights_matrix

def train(dataset):
    '''
    Trains the model, saves the intermediate model states and records loss.
    Parameters:
        dataset (DataLoader): DataLoader instance
    Returns:
        None
    '''

    # if GloVe variation then get the weights matrices
    if 'g' in variation:
        stories_weights_matrix = get_emb_weight_matrix(inp_stories_lang)
        print("stories_weights_matrix.shape: {}".format(stories_weights_matrix.shape))

        questions_weights_matrix = get_emb_weight_matrix(inp_questions_lang)
        print("questions_weights_matrix.shape: {}".format(questions_weights_matrix.shape))

        target_weights_matrix = get_emb_weight_matrix(targ_lang)
        print("target_weights_matrix.shape: {}".format(target_weights_matrix.shape))
    else:
        stories_weights_matrix = None
        questions_weights_matrix = None
        target_weights_matrix = None

    # initialize encoder
    encoder = Encoder(vocab_inp_stories_size, vocab_inp_questions_size, embedding_dim, units, 
                      BATCH_SIZE, img_dim, variation, stories_weights_matrix, questions_weights_matrix)
    # initialize decoder
    decoder = Decoder(vocab_tar_size, embedding_dim, units, units, BATCH_SIZE, variation, target_weights_matrix)

    encoder.to(device)
    decoder.to(device)

    # initialize optimizer 
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), 
                           lr=0.001)

    itr = 0

    losses = []

    for epoch in range(EPOCHS+1):
        start = time.time()
        
        # train
        encoder.train()
        decoder.train()
        
        total_loss = 0
        
        for (batch, inputs) in enumerate(dataset):            
            # if not BERT variation then get the lengths of the stories and questions
            # in case of BERT variation this was taken care of in prepro_data.py
            if 'b' in variation:
                inp_stories, inp_questions, targ, imgs = inputs 
            else:
                inp_stories, inp_questions, targ, inp_stories_len, inp_quetions_len, imgs = inputs
          
            itr = itr + 1
            loss = 0
            
            # input 
            x_stories = inp_stories.transpose(0,1).to(device)
            x_questions = inp_questions.transpose(0,1).to(device)
            
            # target
            ys = targ
            lens = None
            
            enc_output, enc_hidden, imgs_output = encoder(x_stories, x_questions, lens, imgs.to(device), device)
            
            # initialize decoder hidden state with encryption output hidden state
            dec_hidden = enc_hidden
            
            # initialize all the outputs with <start> token 
            dec_input = torch.tensor([[targ_lang.word2idx['<start>']]] * BATCH_SIZE)
            
            # run for every timestep in the batch
            for t in range(1, ys.size(1)):
                predictions, dec_hidden, _ = decoder(dec_input.to(device), dec_hidden.to(device), enc_output.to(device),
                                                imgs_output.to(device))
                loss += loss_function(ys[:, t].to(device), predictions.to(device))
                # using teacher forcing by feeding the true target word as the next input
                dec_input = ys[:, t].unsqueeze(1)
                
            batch_loss = (loss / int(ys.size(1)))
            total_loss += batch_loss
            
            optimizer.zero_grad()
            
            loss.backward()

            # updating model parameters
            optimizer.step()
            
            if batch % 100 == 0:
                print('Epoch {} Iteration {} Batch {} Loss {:.4f}'.format(epoch,
                                                             itr,
                                                             batch,
                                                             batch_loss.detach().item()))
        
        # saving model state at every 20 epochs
        if epoch % 20 == 0:    
            inp_questions_lang, inp_stories_lang
            if inp_questions_lang != None:
                inp_questions_word2idx = inp_questions_lang.word2idx
                inp_stories_word2idx = inp_stories_lang.word2idx
            else:
                inp_questions_word2idx = None
                inp_stories_word2idx = None

            torch.save({
                        "BATCH_SIZE": BATCH_SIZE,
                        "vocab_inp_stories_size": vocab_inp_stories_size,
                        "vocab_inp_questions_size": vocab_inp_questions_size,
                        "vocab_tar_size": vocab_tar_size, 
                        "embedding_dim": embedding_dim,
                        "img_dim": img_dim, 
                        "units": units,
                        "max_length_tar": max_length_tar,
                        "encoder": encoder.state_dict(), 
                        "decoder": decoder.state_dict(),
                        "inp_questions_word2idx": inp_questions_word2idx,
                        "inp_stories_word2idx": inp_stories_word2idx,
                        "targ_word2idx": targ_lang.word2idx,
                        "targ_idx2word": targ_lang.idx2word,
                        "stories_weights_matrix": stories_weights_matrix,
                        "questions_weights_matrix": questions_weights_matrix,
                        "target_weights_matrix": target_weights_matrix

                }, checkpoint_path+"model_epoch-"+str(epoch)+".pt")
        
        print('Epoch {} Iteration {} Loss {:.4f}'.format(epoch, 
                                            itr,
                                            total_loss / N_BATCH))
        losses.append({"epoch": epoch, "loss": str((total_loss / N_BATCH).cpu())})
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    # saving losses
    json.dump(losses, open(checkpoint_path+'losses_'+variation+'.json', 'w'))
    print("---------------------------------")


def load_data(data):
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
    global inp_lang, targ_lang, inp_questions_lang, inp_stories_lang

    original_input_pairs = []

    input_stories = data['input_stories']
    input_questions = data['input_questions']
    
    targets = data['targets']

    # load unique image features
    with h5py.File(input_img_file, 'r') as hf:
        tem = hf.get('images_train')
        uni_image_features = np.array(tem)
    
    tem = data['img_pos']
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
            tem = hf.get('stories_tokens')
            stories_tensor = np.array(tem)

            tem = hf.get('questions_tokens')
            questions_tensor = np.array(tem)
        
            tem = hf.get('unique_tokens')
            unique_tokens = np.array(tem)  

            tem = hf.get('unique_embs')
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
        # create the vocab
        inp_stories_lang = LanguageIndex(data_input_stories)
        # get the indices of tokens in the input story
        input_stories_tensor = [[inp_stories_lang.word2idx[s] for s in item.split(' ')]  for item in data_input_stories]

        # similar to stories, pre-process and get the vocab indices of questions
        data_input_questions = [preprocess_sentence(w) for w in data[:,1]]
        inp_questions_lang = LanguageIndex(data_input_questions)
        input_questions_tensor = [[inp_questions_lang.word2idx[s] for s in item.split(' ')]  for item in data_input_questions]

    # similar to stories, pre-process and get the vocab indices of answer/summary
    data_target = [preprocess_sentence(w) for w in data[:,2]]
    targ_lang = LanguageIndex(data_target)
    target_tensor = [[targ_lang.word2idx[s] for s in item.split(' ')]  for item in data_target]

    print("input_stories_tensor: {}".format(np.array(input_stories_tensor).shape))
    print("input_questions_tensor: {}".format(np.array(input_questions_tensor).shape))
    print("target_tensor: {}".format(np.array(target_tensor).shape))
    print("---------------------------------")

    return data, input_stories_tensor, input_questions_tensor, target_tensor, image_features


EPOCHS = 100
BUFFER_SIZE = 0 
BATCH_SIZE = 0
N_BATCH = 0 
embedding_dim = 0
units = 0
img_dim = 0
vocab_inp_size = 0 
vocab_tar_size = 0 
inp_lang = None 
targ_lang = None 
inp_questions_lang = None
inp_stories_lang = None

glove_embs = None

variation = ''

input_data_file = ''
input_img_file = ''
input_bert_emb = ''

checkpoint_path = ''


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input json
    # variation = isq, iq, sq, bsq, bisq (b - bert embeddings), gisq, gsq (g - glove emb)
    parser.add_argument('--variation', required=True, help='enter variation - isq, iq, sq, bsq, bisq, gisq, gsq')
    parser.add_argument('--input_data_file', required=True, help='enter input data file')
    parser.add_argument('--input_img_file', required=True, help='enter input image file')
    parser.add_argument('--input_bert_emb', default=None, help='enter input bert embeddings file')
    parser.add_argument('--model_save', required=True, help='enter model save path')

    args = parser.parse_args()
    params = vars(args)

    variation = params['variation']
    input_data_file = params['input_data_file']
    input_img_file = params['input_img_file']
    input_bert_emb = params['input_bert_emb']
    checkpoint_path = params['model_save']

    # if GloVe variation then load the pretrained GloVe model using torchtext
    if 'g' in variation:
        glove_embs = torchtext.vocab.GloVe(name="6B", dim=50)

    # load the input train data 
    data_train = json.load(open(input_data_file, 'r'))
    data, input_stories_tensor, input_questions_tensor, target_tensor, input_image_features = load_data(data_train)

    # if any other variation than BERT then pad the stories and questions to maximum length
    # in case of BERT this is already done in prepro_data.py
    if 'b' not in variation:
        max_length_inp_stories, max_length_inp_questions = max_length(input_stories_tensor), max_length(input_questions_tensor) 
        print("max_length_inp_stories: {}".format(max_length_inp_stories))
        print("max_length_inp_questions: {}".format(max_length_inp_questions))
        
        input_stories_tensor = [pad_sequences(x, max_length_inp_stories) for x in input_stories_tensor]
        input_questions_tensor = [pad_sequences(x, max_length_inp_questions) for x in input_questions_tensor]
        print("len(input_stories_tensor): {}".format(len(input_stories_tensor)))
        print("len(input_questions_tensor): {}".format(len(input_questions_tensor)))
    

    max_length_tar = max_length(target_tensor)
    print("max_length_tar: {}".format(max_length_tar))

    target_tensor = [pad_sequences(x, max_length_tar) for x in target_tensor]
    print("len(target_tensor): {}".format(len(target_tensor)))
    print("---------------------------------")


    BUFFER_SIZE = len(input_stories_tensor)
    BATCH_SIZE = 64
    N_BATCH = BUFFER_SIZE//BATCH_SIZE
    img_dim = 4096
    units = 1024

    # set the appropriate parameters as per variation    
    if 'b' in variation:
        embedding_dim = 768
        vocab_inp_size = None
        vocab_inp_stories_size = None
        vocab_inp_questions_size = None
    else:
        if 'g' in variation:
            embedding_dim = 50
        else:
            embedding_dim = 256
        vocab_inp_stories_size = len(inp_stories_lang.word2idx)
        vocab_inp_questions_size = len(inp_questions_lang.word2idx)
        print("vocab_inp_stories_size: {}".format(vocab_inp_stories_size))
        print("vocab_inp_questions_size: {}".format(vocab_inp_questions_size))

    vocab_tar_size = len(targ_lang.word2idx)

    print("vocab_tar_size: {}".format(vocab_tar_size))
    print("---------------------------------")

    train_dataset = PrepareDataForDataLoader(input_stories_tensor, input_questions_tensor, 
                                             target_tensor, input_image_features)

    dataset = DataLoader(train_dataset, batch_size = BATCH_SIZE, 
                         drop_last=True,
                         shuffle=True)

    # train the model    
    train(dataset)
