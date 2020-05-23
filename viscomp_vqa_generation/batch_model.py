import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, img_dim, variation):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.img_dim = img_dim
        self.variation = variation

        if 'i' in self.variation:
            # img_dim = 4096 convert to 256
            self.img_linear = nn.Linear(self.img_dim, self.embedding_dim)

        if 'b' in self.variation:
            # for for already present input embeddings - use linear tranformation
            # vocab_size is input embedding size
            # embeddin_dim is the hidden size
            self.input_dim = self.vocab_size
            self.linear = nn.Linear(self.input_dim, self.embedding_dim)
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        self.gru = nn.GRU(self.embedding_dim, self.enc_units)
        
    def forward(self, x, lens, imgs, device):
        # x: batch_size, max_length 
        
        # print("x size before: {}".format(x.size()))        
        # x: batch_size, max_length, embedding_dim
        if 'b' in self.variation:
            x = x.transpose(0, 1).float()
            x = self.linear(x)   
            x = x.unsqueeze_(-1).permute(2,0,1) 
            # print("x: {}".format(x.shape))
            # x: 1x16x256
        else:
            x = self.embedding(x)

        # print("x size after: {}".format(x.size()))

        
        # print("imgs: {}".format(imgs.shape))
        # imgs: batch_size, 5, 4096

        if 'i' in self.variation:
            imgs = imgs.permute(1,0,2)
            # imgs: 5, batch_size, 4096

            imgs_0 = self.img_linear(imgs[0])
            imgs_1 = self.img_linear(imgs[1])
            imgs_2 = self.img_linear(imgs[2])
            imgs_3 = self.img_linear(imgs[3])
            imgs_4 = self.img_linear(imgs[4])
            # print("imgs_0: {}".format(imgs_0.shape))
            # 16, 256
            imgs = imgs_0 * imgs_1 * imgs_2 * imgs_3 * imgs_4
            # print("imgs: {}".format(imgs.shape))
            # imgs: 16, 256          

            imgs = imgs.unsqueeze_(-1).permute(2,0,1)
            # print("imgs: {}".format(imgs.shape))
            # 1x16x256

            x = x * imgs
            # print("x: {}".format(x.shape))
            # max_len X batch_size X embedding_dim
            # 110x16x256

        if 'b' not in self.variation:
            # x transformed = max_len X batch_size X embedding_dim
            # x = x.permute(1,0,2)
            x = pack_padded_sequence(x, lens) # unpad

        self.hidden = self.initialize_hidden_state(device)
        
        # output: max_length, batch_size, enc_units
        # self.hidden: 1, batch_size, enc_units
        output, self.hidden = self.gru(x, self.hidden) # gru returns hidden state of all timesteps as well as hidden state at last timestep
        
        if 'b' not in self.variation:
            # pad the sequence to the max length in the batch
            output, _ = pad_packed_sequence(output)
        
        return output, self.hidden

    def initialize_hidden_state(self, device):
        return torch.zeros((1, self.batch_sz, self.enc_units)).to(device)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dec_units, enc_units, batch_sz, variation):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.enc_units = enc_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.variation = variation

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim + self.enc_units, 
                          self.dec_units,
                          batch_first=True)
        self.fc = nn.Linear(self.enc_units, self.vocab_size)
        
        # used for attention
        self.W1 = nn.Linear(self.enc_units, self.dec_units)
        self.W2 = nn.Linear(self.enc_units, self.dec_units)
        self.V = nn.Linear(self.enc_units, 1)
    
    def forward(self, x, hidden, enc_output):
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
        
        #score = torch.tanh(self.W2(hidden_with_time_axis) + self.W1(enc_output))
          
        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = torch.softmax(self.V(score), dim=1)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = torch.sum(context_vector, dim=1)
        
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        # takes case of the right portion of the model above (illustrated in red)
        x = self.embedding(x)
        
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        #x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # ? Looks like attention vector in diagram of source
        x = torch.cat((context_vector.unsqueeze(1), x), -1)
        
        # passing the concatenated vector to the GRU
        # output: (batch_size, 1, hidden_size)
        output, state = self.gru(x)
        
        
        # output shape == (batch_size * 1, hidden_size)
        output =  output.view(-1, output.size(2))
        
        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)
        
        return x, state, attention_weights
    
    def initialize_hidden_state(self):
        return torch.zeros((1, self.batch_sz, self.dec_units))
