'''
Adapted from https://github.com/omarsar/pytorch_neural_machine_translation_attention
'''

import torch
import torch.functional as F
import torch.nn as nn


class Encoder(nn.Module):
    ''' 
    Encoder functions.
    '''
    def __init__(self, vocab_stories_size, vocab_questions_size, embedding_dim, enc_units, batch_sz, img_dim, variation, 
                 stories_weights_matrix, questions_weight_matrix):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.vocab_stories_size = vocab_stories_size
        self.vocab_questions_size = vocab_questions_size
        self.embedding_dim = embedding_dim
        self.img_dim = img_dim
        self.variation = variation

        # if images in variation
        if 'i' in self.variation:
            # img_dim = 4096 convert to 1024
            self.img0_linear = nn.Linear(self.img_dim, self.enc_units)
            self.img1_linear = nn.Linear(self.img_dim, self.enc_units)
            self.img2_linear = nn.Linear(self.img_dim, self.enc_units)
            self.img3_linear = nn.Linear(self.img_dim, self.enc_units)
            self.img4_linear = nn.Linear(self.img_dim, self.enc_units)

        # if BERT variation then input dimension = vocab story size
        if 'b' in self.variation:
            self.input_dim = self.vocab_stories_size
        # if GloVe variation then initialze the embedding lookup with the GloVe weights
        elif 'g' in self.variation:
            num_emb, emb_dim = stories_weights_matrix.size()
            self.stories_embedding = torch.nn.Embedding(num_emb, emb_dim)
            self.stories_embedding.load_state_dict({'weight': stories_weights_matrix})
            self.stories_embedding.weight.requires_grad = False

            num_emb, emb_dim = questions_weight_matrix.size()
            self.questions_embedding = torch.nn.Embedding(num_emb, emb_dim)
            self.questions_embedding.load_state_dict({'weight': questions_weight_matrix})
            self.questions_embedding.weight.requires_grad = False
        # if model is to be trained from scratch
        else:
            self.stories_embedding = nn.Embedding(self.vocab_stories_size, self.embedding_dim)
            self.questions_embedding = nn.Embedding(self.vocab_questions_size, self.embedding_dim)

        # story LSTM
        self.lstm_stories = nn.LSTM(self.embedding_dim, self.enc_units)
        self.linear_stories = nn.Linear(2*self.enc_units, self.enc_units)
        
        # question LSTM
        self.lstm_questions = nn.LSTM(self.embedding_dim, self.enc_units)
        self.linear_questions = nn.Linear(2*self.enc_units, self.enc_units)


    def forward(self, x_stories, x_questions, lens, imgs, device): 

        # if images in variation 
        if 'i' in self.variation:
            imgs = imgs.permute(1,0,2)
            # imgs: 5, batch_size, image_dim

            imgs_0 = self.img0_linear(imgs[0])
            imgs_1 = self.img1_linear(imgs[1])
            imgs_2 = self.img2_linear(imgs[2])
            imgs_3 = self.img3_linear(imgs[3])
            imgs_4 = self.img4_linear(imgs[4])
            # (batch_size, enc_units)
            imgs = imgs_0 * imgs_1 * imgs_2 * imgs_3 * imgs_4
            # (1, batch_size, enc_units)
            imgs = imgs.unsqueeze_(-1).permute(2,0,1)
            # initialize the hidden state of LSTM with image features
            x_h0 = imgs
        
        # if images not in variation
        else: 
            # initialize the hidden state of LSTM with zeros
            x_h0 = torch.zeros((1, self.batch_sz, self.enc_units)).to(device)

        
        # if Images+Question variation
        if 'iq' == self.variation:
            x_questions = self.questions_embedding(x_questions)
            x_c0 = torch.zeros((1, self.batch_sz, self.enc_units)).to(device)
            questions_output, (questions_h, questions_c) = self.lstm_questions(x_questions, (x_h0, x_c0))

            return questions_output, questions_h, imgs
        
        # if Images+Story or Story variation
        else:
            # if BERT variation then we already have the corresponding embeddings 
            # and there is no need for lookup  
            if 'b' not in self.variation: 
                x_stories = self.stories_embedding(x_stories)
                
            x_stories_c0 = torch.zeros((1, self.batch_sz, self.enc_units)).to(device)
            stories_output, (stories_h, stories_c) = self.lstm_stories(x_stories, (x_h0, x_stories_c0))
 
           return stories_output, stories_h, imgs

    def initialize_hidden_state(self):
        return torch.zeros((1, self.batch_sz, self.enc_units))


class Decoder(nn.Module):
    '''
    Decoder functions.
    '''
    def __init__(self, vocab_size, embedding_dim, dec_units, enc_units, batch_sz, variation, weights_matrix):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.enc_units = enc_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.variation = variation

        # if GloVe variation then initialize the embedding lookup with the GloVe weights
        if 'g' in variation:
            num_emb, emb_dim = weights_matrix.size()
            self.embedding = torch.nn.Embedding(num_emb, emb_dim)
            self.embedding.load_state_dict({'weight': weights_matrix})
            self.embedding.weight.requires_grad = False
        # if model is to be trained from scratch
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        # GRU
        self.gru = nn.GRU(self.embedding_dim+self.enc_units, self.dec_units, batch_first=True)
        
        # FC layer
        self.fc = nn.Linear(self.enc_units, self.vocab_size)
        
        # attention
        self.W1 = nn.Linear(self.enc_units, self.dec_units)
        self.W2 = nn.Linear(self.enc_units, self.dec_units)
        self.V = nn.Linear(self.enc_units, 1)


    def forward(self, x, hidden, enc_output, imgs):
        
        # attention

        # enc_output original: (max_length, batch_size, enc_units)
        # enc_output converted == (batch_size, max_length, hidden_size)
        enc_output = enc_output.permute(1,0,2)
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score

        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        hidden_with_time_axis = hidden.permute(1, 0, 2)
        
        # score: (batch_size, max_length, hidden_size) # Bahdanaus's
        # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V
        # It doesn't matter which FC we pick for each of the inputs
        score = torch.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = torch.softmax(self.V(score), dim=1)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = torch.sum(context_vector, dim=1)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = torch.cat((context_vector.unsqueeze(1), x), -1)

        # passing the concatenated vector to the GRU
        # output: (batch_size, 1, hidden_size)
        output, state = self.gru(x, hidden)
        
        # output shape == (batch_size * 1, hidden_size)
        output =  output.view(-1, output.size(2))
        
        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)

        return x, state, attention_weights
    
    def initialize_hidden_state(self):
        return torch.zeros((1, self.batch_sz, self.dec_units))
