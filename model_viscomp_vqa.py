#-*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os, h5py, sys, argparse
import ipdb
import time
import math
import cv2
import codecs, json
from tensorflow.compat.v1.nn import rnn_cell
from sklearn.metrics import average_precision_score
import gc
import sys

class Answer_Generator():
    def __init__(self, rnn_size, rnn_layer, batch_size, input_embedding_size, dim_image, dim_hidden, 
        max_words_q, vocabulary_size, max_words_d, vocabulary_size_d, drop_out_rate, num_answers, 
        variation, offline_text):
        self.rnn_size = rnn_size
        self.rnn_layer = rnn_layer
        self.batch_size = batch_size
        self.input_embedding_size = input_embedding_size
        self.dim_image = dim_image
        self.dim_hidden = dim_hidden
        self.max_words_q = max_words_q
        self.vocabulary_size = vocabulary_size
        self.max_words_d = max_words_d
        self.vocabulary_size_d = vocabulary_size_d	
        self.drop_out_rate = drop_out_rate
        self.num_answers = num_answers
        self.variation = variation
        self.offline_text = offline_text

    	# question-embedding
        self.embed_ques_W = tf.Variable(tf.random.uniform([self.vocabulary_size, self.input_embedding_size], -0.08, 0.08), name='embed_ques_W')

    	# encoder: RNN body
        self.lstm_1 = rnn_cell.LSTMCell(rnn_size, input_embedding_size, state_is_tuple=False)
        self.lstm_dropout_1 = rnn_cell.DropoutWrapper(self.lstm_1, output_keep_prob = 1 - self.drop_out_rate)
        self.lstm_2 = rnn_cell.LSTMCell(rnn_size, rnn_size, state_is_tuple=False)
        self.lstm_dropout_2 = rnn_cell.DropoutWrapper(self.lstm_2, output_keep_prob = 1 - self.drop_out_rate)
        self.stacked_lstm = rnn_cell.MultiRNNCell([self.lstm_dropout_1, self.lstm_dropout_2])

    	# state-embedding
        self.embed_state_W = tf.Variable(tf.random.uniform([2*rnn_size*rnn_layer, self.dim_hidden], -0.08,0.08),name='embed_state_W')
        self.embed_state_b = tf.Variable(tf.random.uniform([self.dim_hidden], -0.08, 0.08), name='embed_state_b')
    	
        if self.variation in ['isq', 'sq']:
            if self.offline_text == "False":
                # print("\n\n\noffline_text false\n\n\n")
                # description-embedding
                self.embed_desc_W = tf.Variable(tf.random.uniform([self.vocabulary_size_d, self.input_embedding_size], -0.08, 0.08), name='embed_desc_W')

                # encoder: RNN body
                self.lstm_1_d = rnn_cell.LSTMCell(rnn_size, input_embedding_size, state_is_tuple=False)
                self.lstm_dropout_1_d = rnn_cell.DropoutWrapper(self.lstm_1_d, output_keep_prob = 1 - self.drop_out_rate)
                self.lstm_2_d = rnn_cell.LSTMCell(rnn_size, rnn_size, state_is_tuple=False)
                self.lstm_dropout_2_d = rnn_cell.DropoutWrapper(self.lstm_2_d, output_keep_prob = 1 - self.drop_out_rate)
                self.stacked_lstm_d = rnn_cell.MultiRNNCell([self.lstm_dropout_1_d, self.lstm_dropout_2_d])

                # description state-embedding
                self.embed_state_desc_W = tf.Variable(tf.random.uniform([2*rnn_size*rnn_layer, self.dim_hidden], -0.08,0.08),name='embed_state_desc_W')
            elif self.offline_text == "True":
                # print("\n\n\noffline_text true\n\n\n")
                self.embed_state_desc_W = tf.Variable(tf.random.uniform([self.dim_hidden, self.dim_hidden], -0.08,0.08),name='embed_state_desc_W')   

            self.embed_state_desc_b = tf.Variable(tf.random.uniform([self.dim_hidden], -0.08, 0.08), name='embed_state_desc_b')


        if self.variation in ['isq', 'iq']:
            # image-embedding 1
            self.embed_image_W = tf.Variable(tf.random.uniform([dim_image, self.dim_hidden], -0.08, 0.08), name='embed_image_W')
            self.embed_image_b = tf.Variable(tf.random.uniform([dim_hidden], -0.08, 0.08), name='embed_image_b')

            # my code
            # image-embedding 2
            self.embed_image2_W = tf.Variable(tf.random.uniform([dim_image, self.dim_hidden], -0.08, 0.08), name='embed_image2_W')
            self.embed_image2_b = tf.Variable(tf.random.uniform([dim_hidden], -0.08, 0.08), name='embed_image2_b')

            # image-embedding 3
            self.embed_image3_W = tf.Variable(tf.random.uniform([dim_image, self.dim_hidden], -0.08, 0.08), name='embed_image3_W')
            self.embed_image3_b = tf.Variable(tf.random.uniform([dim_hidden], -0.08, 0.08), name='embed_image3_b')

            # image-embedding 4
            self.embed_image4_W = tf.Variable(tf.random.uniform([dim_image, self.dim_hidden], -0.08, 0.08), name='embed_image4_W')
            self.embed_image4_b = tf.Variable(tf.random.uniform([dim_hidden], -0.08, 0.08), name='embed_image4_b')

            # image-embedding 5
            self.embed_image5_W = tf.Variable(tf.random.uniform([dim_image, self.dim_hidden], -0.08, 0.08), name='embed_image5_W')
            self.embed_image5_b = tf.Variable(tf.random.uniform([dim_hidden], -0.08, 0.08), name='embed_image5_b')


    	# options-embedding
        self.embed_options_W = tf.Variable(tf.random.uniform([self.num_answers, options_embedding_size], -0.1, 0.1), name='embed_options_W')
        # print("\n\nself.embed_options_W: {}\n\n".format(self.embed_options_W))

        # self.lstm_o = rnn_cell.LSTMCell(64, state_is_tuple=False, reuse=tf.AUTO_REUSE)

        # self.lstm_1_o = rnn_cell.LSTMCell(rnn_size, input_embedding_size, state_is_tuple=False)
        # self.lstm_dropout_1_o = rnn_cell.DropoutWrapper(self.lstm_1_o, output_keep_prob = 1 - self.drop_out_rate)
        # self.lstm_2_o = rnn_cell.LSTMCell(rnn_size, rnn_size, state_is_tuple=False)
        # self.lstm_dropout_2_o = rnn_cell.DropoutWrapper(self.lstm_2_o, output_keep_prob = 1 - self.drop_out_rate)
        # self.stacked_lstm_o = rnn_cell.MultiRNNCell([self.lstm_dropout_1_o, self.lstm_dropout_2_o])

        # options state-embedding
        # self.embed_options_state_W = tf.Variable(tf.random.uniform([2*rnn_size*rnn_layer, self.dim_hidden], -0.08,0.08),name='embed_options_state_W')
        # self.embed_options_b = tf.Variable(tf.random.uniform([self.dim_hidden], -0.08, 0.08), name='embed_options_b')

        # end my code

        # score-embedding


        self.embed_score_W = tf.Variable(tf.random.uniform([dim_hidden, options_embedding_size], -0.08, 0.08), name='embed_score_W')
        self.embed_h_b = tf.Variable(tf.random.uniform([options_embedding_size], -0.08, 0.08), name='embed_h_b')
        self.embed_score_b = tf.Variable(tf.random.uniform([num_output], -0.08, 0.08), name='embed_score_b')

        # score-embedding
        # self.embed_score_W = tf.Variable(tf.random.uniform([128, 5], -0.08, 0.08), name='embed_score_W')
        # self.embed_score_b = tf.Variable(tf.random.uniform([5], -0.08, 0.08), name='embed_score_b')

    def build_model(self):
        if self.variation in ['isq', 'iq']:
            image = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.dim_image])
            
            # my code
            image2 = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.dim_image])
            image3 = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.dim_image])
            image4 = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.dim_image])
            image5 = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.dim_image])
            # end my code

        question = tf.compat.v1.placeholder(tf.int32, [self.batch_size, self.max_words_q])
        
        if self.variation in ['isq', 'sq']:
            if self.offline_text == "False":
                # print("\n\n\noffline_text false\n\n\n")
                description = tf.compat.v1.placeholder(tf.int32, [self.batch_size, self.max_words_d])
            else:
                # print("\n\n\noffline_text true\n\n\n")
                description_embedding = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.dim_hidden])

        mc_answers = tf.compat.v1.placeholder(tf.int32, [self.batch_size, num_output])

        label = tf.compat.v1.placeholder(tf.int64, [self.batch_size,]) 
        loss = 0.0

        # question embed
        # print("self.stacked_lstm.state_size: {}".format(self.stacked_lstm.state_size))
        # state = tf.zeros([self.batch_size, self.stacked_lstm.state_size])
        state = self.stacked_lstm.zero_state(self.batch_size, dtype=tf.float32)
        for i in range(max_words_q):
            if i==0:
                # print("self.input_embedding_size: {}".format(self.input_embedding_size))
                ques_emb_linear = tf.zeros([self.batch_size, self.input_embedding_size])
            else:
                # print("hello")
                tf.get_variable_scope().reuse_variables()
                ques_emb_linear = tf.nn.embedding_lookup(self.embed_ques_W, question[:,i-1])


            ques_emb_drop = tf.nn.dropout(ques_emb_linear, 1-self.drop_out_rate)
            ques_emb = tf.tanh(ques_emb_drop)

            output, state = self.stacked_lstm(ques_emb, state)

    	# multimodal (fusing question & image)
        state_drop = tf.nn.dropout(state, 1-self.drop_out_rate)
        # print("\n\n\n\n\nweight: {} \n bias: {} \n\n\n\n\n\n".format(self.embed_state_W, self.embed_state_b))
        state_linear = tf.compat.v1.nn.xw_plus_b(tf.concat([state_drop[0], state_drop[1]], 1), self.embed_state_W, self.embed_state_b)
        state_emb = tf.tanh(state_linear)

        if self.variation in ['isq', 'sq']:
            if self.offline_text == "False":
                # print("\n\n\noffline_text false\n\n\n")
                # description embed
                state_d = self.stacked_lstm_d.zero_state(self.batch_size, dtype=tf.float32)
                for i in range(max_words_d):
                    if i==0:
                        # print("self.input_embedding_size: {}".format(self.input_embedding_size))
                        desc_emb_linear = tf.zeros([self.batch_size, self.input_embedding_size])
                    else:
                        # print("hello")
                        tf.get_variable_scope().reuse_variables()
                        desc_emb_linear = tf.nn.embedding_lookup(self.embed_desc_W, description[:,i-1])


                    desc_emb_drop = tf.nn.dropout(desc_emb_linear, 1-self.drop_out_rate)
                    desc_emb = tf.tanh(desc_emb_drop)

                    output_d, state_d = self.stacked_lstm_d(desc_emb, state_d)

                # description
                state_drop_d = tf.nn.dropout(state_d, 1-self.drop_out_rate)
                # print("\nstate0: {} \n state1: {} \n".format(state_drop[0], state_drop[1]))
                state_linear_d = tf.compat.v1.nn.xw_plus_b(tf.concat([state_drop_d[0], state_drop_d[1]], 1), self.embed_state_desc_W, self.embed_state_desc_b)
                state_emb_d = tf.tanh(state_linear_d)
            else:
                # print("\n\n\noffline_text true\n\n\n")
                state_drop_d = tf.nn.dropout(description_embedding, 1-self.drop_out_rate)
                state_linear_d = tf.compat.v1.nn.xw_plus_b(state_drop_d, self.embed_state_desc_W, self.embed_state_desc_b)
                state_emb_d = tf.tanh(state_linear_d)

        if self.variation in ['isq', 'iq']:
            # images
            image_drop = tf.nn.dropout(image, 1-self.drop_out_rate)
            image_linear = tf.compat.v1.nn.xw_plus_b(image_drop, self.embed_image_W, self.embed_image_b)
            # image_emb = tf.tanh(image_linear)

            # my code
            image2_drop = tf.nn.dropout(image2, 1-self.drop_out_rate)
            image2_linear = tf.compat.v1.nn.xw_plus_b(image2_drop, self.embed_image2_W, self.embed_image2_b)
            image3_drop = tf.nn.dropout(image3, 1-self.drop_out_rate)
            image3_linear = tf.compat.v1.nn.xw_plus_b(image3_drop, self.embed_image3_W, self.embed_image3_b)
            image4_drop = tf.nn.dropout(image4, 1-self.drop_out_rate)
            image4_linear = tf.compat.v1.nn.xw_plus_b(image4_drop, self.embed_image4_W, self.embed_image4_b)
            image5_drop = tf.nn.dropout(image5, 1-self.drop_out_rate)
            image5_linear = tf.compat.v1.nn.xw_plus_b(image5_drop, self.embed_image5_W, self.embed_image5_b)

            # final_image_linear = tf.multiply(image_linear, image2_linear)
            final_image_linear = image_linear * image2_linear * image3_linear * image4_linear * image5_linear
            image_emb = tf.tanh(final_image_linear)        

        # options answers

        options_list = []

        for i in range(0, num_output):
            # option_state = self.lstm_o.zero_state(self.batch_size, dtype=tf.float32)
            tf.get_variable_scope().reuse_variables()
            option_emb_linear = tf.nn.embedding_lookup(self.embed_options_W, mc_answers[:,i])
            # print("\n\noption_emb_linear: {}\n\n".format(option_emb_linear))
            # option_emb_drop = tf.nn.dropout(option_emb_linear, 1-self.drop_out_rate)
            # option_emb = tf.tanh(option_emb_drop)
            # _, option_state = self.lstm_o(option_emb_linear, option_state)
            options_list.append(option_emb_linear)     

        options_r_vector = tf.stack(options_list)
        options_r_vector = tf.transpose(options_r_vector, perm=[1, 0, 2])

        # my code end

        if self.variation == 'isq':
            # scores = tf.multiply(state_emb, image_emb)
            h = state_emb * image_emb * state_emb_d
        elif self.variation == 'iq':
            h = state_emb * image_emb
        elif self.variation == 'sq':
            h = state_emb * state_emb_d
        else:
            h = state_emb

        h_drop = tf.nn.dropout(h, 1-self.drop_out_rate)
        # print("\nh_drop: {} \nembed_score_W: {} \noptions_r_vector: {}".format(h_drop, self.embed_score_W, options_r_vector))
        score_emb_h = tf.compat.v1.nn.xw_plus_b(h_drop, self.embed_score_W, self.embed_h_b)
        # score_emb_h = tf.matmul(h_drop, self.embed_score_W)
        # print("\n\n\nscore_emb_h: {} \n\n\n".format(score_emb_h))

        repeats = [num_output, 1]
        expanded_tensor = tf.expand_dims(score_emb_h, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        score_emb_h = tf.reshape(tiled_tensor, tf.shape(score_emb_h) * repeats)

        # print("\n\n\nscore_emb_h repeat: {} \n\n\n".format(score_emb_h))
        options_r_vector = tf.reshape(options_r_vector, (self.batch_size*num_output, options_embedding_size))
        # print("\n\n\noptions_r_vector reshape: {} \n\n\n".format(options_r_vector))
        scores_emb = tf.einsum('ij,ij->i', options_r_vector, score_emb_h)
        scores_emb = scores_emb + tf.tile(self.embed_score_b, [self.batch_size])
        scores_emb = tf.reshape(scores_emb, (self.batch_size, num_output))
        # print("\n\n\nscore_emb: {} \n\n\n".format(scores_emb))

    	# Calculate cross entropy
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores_emb, labels=label)
        # Calculate loss
        loss = tf.reduce_mean(cross_entropy)
        # print("\n\nloss: {} \n\n".format(loss))

        # score_emb = tf.compat.v1.nn.xw_plus_b(options_r_vector, tf.transpose(self.embed_score_W), h_drop)

        # print("\n\nself.embed_options_W: {}\n\n".format(self.embed_options_W))
        if self.variation == 'isq':
            # print("\n\n\n\n\n\n\nisq\n\n\n\n\n\n\n")
            if self.offline_text == "False":
                # print("\n\n\noffline_text false\n\n\n")
                return loss, image, image2, image3, image4, image5, description, question, mc_answers, label   # my code
            else:
                # print("\n\n\noffline_text true\n\n\n")
                # print("\n\n\n\ndescription_embedding: {}\n\n\n\n".format(description_embedding))
                return loss, image, image2, image3, image4, image5, description_embedding, question, mc_answers, label   # my code
        elif self.variation == 'iq':
            # print("\n\n\n\n\n\n\niq\n\n\n\n\n\n\n")
            return loss, image, image2, image3, image4, image5, question, mc_answers, label   # my code
        elif self.variation == 'sq':
            # print("\n\n\n\n\n\n\nsq\n\n\n\n\n\n\n")
            if self.offline_text == "False":
                return loss, description, question, mc_answers, label   # my code
            else:
                return loss, description_embedding, question, mc_answers, label   # my code
        else:
            # print("\n\n\n\n\n\n\nq\n\n\n\n\n\n\n")
            return loss, question, mc_answers, label   # my code
        # return loss, image, question, label
    
    def build_generator(self):
        # print("\n\n\n\nvariation: {}\n\n\n\n".format(variation))
        if self.variation in ['isq', 'iq']:
            image = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.dim_image])
            # my code
            image2 = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.dim_image])
            image3 = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.dim_image])
            image4 = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.dim_image])
            image5 = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.dim_image])
            # my code end

        question = tf.compat.v1.placeholder(tf.int32, [self.batch_size, self.max_words_q])
        
        if self.variation in ['isq', 'sq']:
            if self.offline_text == "False":
                description = tf.compat.v1.placeholder(tf.int32, [self.batch_size, self.max_words_d])
            else:
                description_embedding = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.dim_hidden])

        mc_answers = tf.compat.v1.placeholder(tf.int32, [self.batch_size, num_output])

        # state = tf.zeros([self.batch_size, self.stacked_lstm.state_size])
        state = self.stacked_lstm.zero_state(self.batch_size, dtype=tf.float32)
        loss = 0.0
        for i in range(max_words_q):
            if i==0:
                ques_emb_linear = tf.zeros([self.batch_size, self.input_embedding_size])
            else:
                tf.get_variable_scope().reuse_variables()
                ques_emb_linear = tf.nn.embedding_lookup(self.embed_ques_W, question[:,i-1])

            ques_emb_drop = tf.nn.dropout(ques_emb_linear, 1-self.drop_out_rate)
            ques_emb = tf.tanh(ques_emb_drop)

            output, state = self.stacked_lstm(ques_emb, state)

        # multimodal (fusing question & image)
        # question
        state_drop = tf.nn.dropout(state, 1-self.drop_out_rate)
        # state_linear = tf.nn.xw_plus_b(state_drop, self.embed_state_W, self.embed_state_b)
        state_linear = tf.compat.v1.nn.xw_plus_b(tf.concat([state_drop[0], state_drop[1]], 1), self.embed_state_W, self.embed_state_b)
        state_emb = tf.tanh(state_linear)
    	
        if self.variation in ['isq', 'sq']:
            if self.offline_text == "False":
                # description embed
                state_d = self.stacked_lstm_d.zero_state(self.batch_size, dtype=tf.float32)
                for i in range(max_words_d):
                    if i==0:
                        # print("self.input_embedding_size: {}".format(self.input_embedding_size))
                        desc_emb_linear = tf.zeros([self.batch_size, self.input_embedding_size])
                    else:
                        # print("hello")
                        tf.get_variable_scope().reuse_variables()
                        desc_emb_linear = tf.nn.embedding_lookup(self.embed_desc_W, description[:,i-1])


                    desc_emb_drop = tf.nn.dropout(desc_emb_linear, 1-self.drop_out_rate)
                    desc_emb = tf.tanh(desc_emb_drop)

                    output_d, state_d = self.stacked_lstm_d(desc_emb, state_d)

                # description
                state_drop_d = tf.nn.dropout(state_d, 1-self.drop_out_rate)
                # print("\nstate0: {} \n state1: {} \n".format(state_drop[0], state_drop[1]))
                state_linear_d = tf.compat.v1.nn.xw_plus_b(tf.concat([state_drop_d[0], state_drop_d[1]], 1), self.embed_state_desc_W, self.embed_state_desc_b)
                state_emb_d = tf.tanh(state_linear_d)
            else:
                state_drop_d = tf.nn.dropout(description_embedding, 1-self.drop_out_rate)
                state_linear_d = tf.compat.v1.nn.xw_plus_b(state_drop_d, self.embed_state_desc_W, self.embed_state_desc_b)
                state_emb_d = tf.tanh(state_linear_d)

        if self.variation in ['isq', 'iq']:
            # images
            image_drop = tf.nn.dropout(image, 1-self.drop_out_rate)
            image_linear = tf.compat.v1.nn.xw_plus_b(image_drop, self.embed_image_W, self.embed_image_b)
            # image_emb = tf.tanh(image_linear)

            # my code
            image2_drop = tf.nn.dropout(image2, 1-self.drop_out_rate)
            image2_linear = tf.compat.v1.nn.xw_plus_b(image2_drop, self.embed_image2_W, self.embed_image2_b)
            image3_drop = tf.nn.dropout(image3, 1-self.drop_out_rate)
            image3_linear = tf.compat.v1.nn.xw_plus_b(image3_drop, self.embed_image3_W, self.embed_image3_b)
            image4_drop = tf.nn.dropout(image4, 1-self.drop_out_rate)
            image4_linear = tf.compat.v1.nn.xw_plus_b(image4_drop, self.embed_image4_W, self.embed_image4_b)
            image5_drop = tf.nn.dropout(image5, 1-self.drop_out_rate)
            image5_linear = tf.compat.v1.nn.xw_plus_b(image5_drop, self.embed_image5_W, self.embed_image5_b)

            # final_image_linear = tf.multiply(image_linear, image2_linear)
            final_image_linear = image_linear * image2_linear * image3_linear * image4_linear * image5_linear
            image_emb = tf.tanh(final_image_linear)

        options_list = []

        for i in range(0, num_output):
            # option_state = self.lstm_o.zero_state(self.batch_size, dtype=tf.float32)
            tf.get_variable_scope().reuse_variables()
            option_emb_linear = tf.nn.embedding_lookup(self.embed_options_W, mc_answers[:,i])
            # option_emb_drop = tf.nn.dropout(option_emb_linear, 1-self.drop_out_rate)
            # option_emb = tf.tanh(option_emb_drop)
            # _, option_state = self.lstm_o(option_emb_linear, option_state)
            options_list.append(option_emb_linear)     

        options_r_vector = tf.stack(options_list)
        options_r_vector = tf.transpose(options_r_vector, perm=[1, 0, 2])

        # my code end

        if self.variation == 'isq':
            # scores = tf.multiply(state_emb, image_emb)
            h = state_emb * image_emb * state_emb_d
        elif self.variation == 'iq':
            h = state_emb * image_emb
        elif self.variation == 'sq':
            h = state_emb * state_emb_d
        else:
            h = state_emb

        h_drop = tf.nn.dropout(h, 1-self.drop_out_rate)
        # print("\nh_drop: {} \nembed_score_W: {} \noptions_r_vector: {}".format(h_drop, self.embed_score_W, options_r_vector))
        score_emb_h = tf.compat.v1.nn.xw_plus_b(h_drop, self.embed_score_W, self.embed_h_b)
        # print("\n\n\nscore_emb_h: {} \n\n\n".format(score_emb_h))

        repeats = [num_output, 1]
        expanded_tensor = tf.expand_dims(score_emb_h, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        score_emb_h = tf.reshape(tiled_tensor, tf.shape(score_emb_h) * repeats)

        # print("\n\n\nscore_emb_h repeat: {} \n\n\n".format(score_emb_h))
        options_r_vector = tf.reshape(options_r_vector, (self.batch_size*num_output, options_embedding_size))
        # print("\n\n\noptions_r_vector reshape: {} \n\n\n".format(options_r_vector))
        scores_emb = tf.einsum('ij,ij->i', options_r_vector, score_emb_h)
        scores_emb = scores_emb + tf.tile(self.embed_score_b, [self.batch_size]) 
        scores_emb = tf.reshape(scores_emb, (self.batch_size, num_output))
        # print("\n\n\nscore_emb: {} \n\n\n".format(scores_emb))

        # my code end

        # scores = tf.multiply(state_emb, image_emb)
        # scores = state_emb * image_emb * state_emb_d * options_state_emb
        # scores_drop = tf.nn.dropout(scores, 1-self.drop_out_rate)
        # scores_emb = tf.compat.v1.nn.xw_plus_b(scores_drop, self.embed_scor_W, self.embed_scor_b) 

        # FINAL ANSWER
        # generated_ANS = tf.compat.v1.nn.xw_plus_b(scores_drop, self.embed_scor_W, self.embed_scor_b)
        generated_ANS = scores_emb

        if self.variation == 'isq':
            if self.offline_text == "False":
                return generated_ANS, image, image2, image3, image4, image5, description, question, mc_answers   # my code
            else:
                # print("\n\n\n\ndescription_embedding: {}\n\n\n\n".format(description_embedding))
                return generated_ANS, image, image2, image3, image4, image5, description_embedding, question, mc_answers   # my code
        elif self.variation == 'iq':
            return generated_ANS, image, image2, image3, image4, image5, question, mc_answers   # my code
        elif self.variation == 'sq':
            if self.offline_text == "False":
                return generated_ANS, description, question, mc_answers   # my code
            else:
                return generated_ANS, description_embedding, question, mc_answers   # my code
        else:
            return generated_ANS, question, mc_answers
        # return generated_ANS, image, image2, image3, image4, image5, description, question, mc_answers # my code
        # return generated_ANS, image, question
    
#####################################################
#                 Global Parameters		    #  
#####################################################
print('Loading parameters ...')
# Data input setting
input_img_h5 = './data_img.h5'
input_ques_h5 = './data_prepro.h5'
input_json = './data_prepro.json'

input_text_h5 = './data_text.h5'

# Train Parameters setting
learning_rate = 0.0003			# learning rate for rmsprop
#starter_learning_rate = 3e-4
learning_rate_decay_start = -1		# at what iteration to start decaying learning rate? (-1 = dont)
# batch_size = 500			# batch_size for each iterations
batch_size = 30
input_embedding_size = 200		# he encoding size of each token in the vocabulary
rnn_size = 512				# size of the rnn in number of hidden nodes in each layer
rnn_layer = 2				# number of the rnn layer
dim_image = 4096
dim_hidden = 1024 #1024			# size of the common embedding vector


num_output = 5			# number of output answers


img_norm = 1				# normalize the image feature. 1 = normalize, 0 = not normalize
decay_factor = 0.99997592083

variation = ''
offline_text = None

# Check point
checkpoint_path = ''

# misc
gpu_id = 0
# max_itr = 150000
max_itr = 1200  # 30000
n_epochs = 1  # 3000
max_words_q = 15
max_words_d = 50

# num_answers = 0
# num_answer = 131
options_embedding_size = 128


#####################################################

def right_align(seq,lengths):
    v = np.zeros(np.shape(seq))
    N = np.shape(seq)[1]
    for i in range(np.shape(seq)[0]):
        v[i][N-lengths[i]:N-1]=seq[i][0:lengths[i]-1]
    return v

def get_data():

    dataset = {}
    train_data = {}
    # load json file
    print('loading json file...')
    with open(input_json) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    if variation in ['isq', 'iq']:
        # load image feature
        print('loading image feature...')
        with h5py.File(input_img_h5,'r') as hf:
            # -----0~82459------
            tem = hf.get('images_train')
            img_feature = np.array(tem)
    
    # load h5 file
    print('loading h5 file...')
    with h5py.File(input_ques_h5,'r') as hf:
        # total number of answers
        train_data['num_answers'] = np.array(hf.get('num_answers'))
        # print("\n\n\nnum_answers: {}\n\n\n".format(train_data['num_answers']))
        # total number of training data is 215375
        # question is (26, )
        tem = hf.get('ques_train')
        train_data['question'] = np.array(tem)-1
        # max length is 23
        tem = hf.get('ques_length_train')
        train_data['length_q'] = np.array(tem)

        if variation in ['isq', 'sq']:
            if offline_text == "False":
                # question is 
                tem = hf.get('description_train')
                train_data['description'] = np.array(tem)-1
                # max length is 
                tem = hf.get('description_length_train')
                train_data['length_d'] = np.array(tem)
            else:
                with h5py.File(input_text_h5,'r') as hf_text:
                    tem = hf_text.get('text_train')
                    train_data['description_embedding'] = np.array(tem)

        if variation in ['isq', 'iq']:
            # total 82460 img
            tem = hf.get('img_pos_train')
    	    # convert into 0~82459
            train_data['img_list'] = np.array(tem)-1
        
        # answer is 1~1000
        tem = hf.get('answers')
        train_data['answers'] = np.array(tem)-1

        # multiple choice options
        tem = hf.get('MC_ans_train')
        train_data['MC_ans_train'] = np.array(tem)-1

    print('question aligning')
    train_data['question'] = right_align(train_data['question'], train_data['length_q'])

    if variation == 'q':
        return dataset, train_data

    if variation in ['isq', 'sq']:
        if offline_text == "False":
            print('description aligning')
            train_data['description'] = right_align(train_data['description'], train_data['length_d'])
        
        if variation == 'sq':
            return dataset, train_data

    if variation in ['isq', 'iq']:
        print('Normalizing image feature')
        if img_norm:
            tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature), axis=1))
            print('tem: {}'.format(tem.shape))
            print('image feature dim: {}'.format(img_feature.shape))
            img_feature = np.divide(img_feature, np.transpose(np.tile(tem,(4096,1))))

        return dataset, img_feature, train_data


def train():
    print ('loading dataset...')
    if variation in ['isq', 'iq']:
        dataset, img_feature, train_data = get_data()
    else:
        dataset, train_data = get_data()
    
    num_train = train_data['question'].shape[0]
    # print("\n\n\n\n number of train instances: {} \n\n\n\n".format(num_train))
    vocabulary_size = len(dataset['ix_to_word'].keys())
    print ('vocabulary_size : ' + str(vocabulary_size))

    if variation in ['isq', 'sq']:
        vocabulary_size_d = len(dataset['ix_to_word_d'].keys()) 
        print ('vocabulary_size_d : ' + str(vocabulary_size_d))
    else:
        vocabulary_size_d = 0

    print ('constructing  model...')
    model = Answer_Generator(
            rnn_size = rnn_size,
            rnn_layer = rnn_layer,
            batch_size = batch_size,
            input_embedding_size = input_embedding_size,
            dim_image = dim_image,
            dim_hidden = dim_hidden,
            max_words_q = max_words_q,	
            vocabulary_size = vocabulary_size,
            max_words_d = max_words_d,   
            vocabulary_size_d = vocabulary_size_d,  
            drop_out_rate = 0.5,
            num_answers = train_data['num_answers'],
            variation = variation,
            offline_text = offline_text)

    # tf_loss, tf_image, tf_question, tf_label = model.build_model()
    if variation == 'isq':
        if offline_text == "False":
            # print("\n\n\noffline_text false\n\n\n")
            tf_loss, tf_image, tf_image2, tf_image3, tf_image4, tf_image5, tf_description, tf_question, tf_mc_answers, tf_label = model.build_model() # my code
        else:
            # print("\n\n\noffline_text true\n\n\n")
            tf_loss, tf_image, tf_image2, tf_image3, tf_image4, tf_image5, tf_description_embedding, tf_question, tf_mc_answers, tf_label = model.build_model() # my code
    elif variation == 'iq':
        tf_loss, tf_image, tf_image2, tf_image3, tf_image4, tf_image5, tf_question, tf_mc_answers, tf_label = model.build_model() # my code
    elif variation == 'sq':
        if offline_text == "False":
            tf_loss, tf_description, tf_question, tf_mc_answers, tf_label = model.build_model() # my code
        else:
            tf_loss, tf_description_embedding, tf_question, tf_mc_answers, tf_label = model.build_model() # my code
    else:
        tf_loss, tf_question, tf_mc_answers, tf_label = model.build_model() # my code

    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver(max_to_keep=100)

    tvars = tf.compat.v1.trainable_variables()
    lr = tf.Variable(learning_rate)
    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
    # gradient clipping
    gvs = opt.compute_gradients(tf_loss,tvars)
    clipped_gvs = [(tf.clip_by_value(grad, -10.0, 10.0), var) for grad, var in gvs]
    # tf.get_variable_scope().reuse_variables()
    # with tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:
    #     assert tf.get_variable_scope().reuse == True
    with tf.compat.v1.variable_scope('embed_ques_W/Adam', reuse=tf.compat.v1.AUTO_REUSE) as scope:
        train_op = opt.apply_gradients(clipped_gvs)

    # tf.initialize_all_variables().run()
    tf.compat.v1.global_variables_initializer().run()

    losses = []  #my code

    print ('start training...')
    tStart_total = time.time()
    for itr in range(max_itr+1):
        tStart = time.time()
        # shuffle the training data
        index = np.random.randint(0, num_train-1, batch_size)

        current_question = train_data['question'][index,:]
        current_length_q = train_data['length_q'][index]

        if variation in ['isq', 'sq']:
            if offline_text == "False":
                # print("\n\n\noffline_text false\n\n\n")
                current_description = train_data['description'][index,:]
                current_length_d = train_data['length_d'][index]
            else:
                # print("\n\n\noffline_text true\n\n\n")
                current_description_embedding = train_data['description_embedding'][index,:]

        current_answers = train_data['answers'][index]

        if variation in ['isq', 'iq']:
            current_img_list = train_data['img_list'][index][:,0]
            current_img = img_feature[current_img_list,:]

            # my code
            current_img2_list = train_data['img_list'][index][:,1]
            current_img2 = img_feature[current_img2_list,:]
            current_img3_list = train_data['img_list'][index][:,2]
            current_img3 = img_feature[current_img3_list,:]
            current_img4_list = train_data['img_list'][index][:,3]
            current_img4 = img_feature[current_img4_list,:]
            current_img5_list = train_data['img_list'][index][:,4]
            current_img5 = img_feature[current_img5_list,:]

        current_mc_answers = train_data['MC_ans_train'][index,:]
        # print("\n\ncurrent_mc_answers: {} \n\n".format(current_mc_answers))
        current_length_mc_answers = num_output
        # my code end

        # do the training process!!!
        if variation == 'isq':
            if offline_text == "False":
                # print("\n\n\noffline_text false\n\n\n")
                _, loss = sess.run(
                        [train_op, tf_loss],
                        feed_dict={
                            tf_image: current_img,
                            tf_image2: current_img2,  # my code
                            tf_image3: current_img3,  # my code
                            tf_image4: current_img4,  # my code
                            tf_image5: current_img5,  # my code
                            tf_description: current_description,
                            tf_question: current_question,
                            tf_mc_answers: current_mc_answers,
                            tf_label: current_answers
                            })
            else:
                # print("\n\n\noffline_text true\n\n\n")
                _, loss = sess.run(
                        [train_op, tf_loss],
                        feed_dict={
                            tf_image: current_img,
                            tf_image2: current_img2,  # my code
                            tf_image3: current_img3,  # my code
                            tf_image4: current_img4,  # my code
                            tf_image5: current_img5,  # my code
                            tf_description_embedding: current_description_embedding,
                            tf_question: current_question,
                            tf_mc_answers: current_mc_answers,
                            tf_label: current_answers
                            })
        elif variation == 'iq':
            _, loss = sess.run(
                    [train_op, tf_loss],
                    feed_dict={
                        tf_image: current_img,
                        tf_image2: current_img2,  # my code
                        tf_image3: current_img3,  # my code
                        tf_image4: current_img4,  # my code
                        tf_image5: current_img5,  # my code
                        tf_question: current_question,
                        tf_mc_answers: current_mc_answers,
                        tf_label: current_answers
                        })
        elif variation == 'sq':
            if offline_text == "False":
                _, loss = sess.run(
                        [train_op, tf_loss],
                        feed_dict={
                            tf_description: current_description,
                            tf_question: current_question,
                            tf_mc_answers: current_mc_answers,
                            tf_label: current_answers
                            })
            else:
                _, loss = sess.run(
                        [train_op, tf_loss],
                        feed_dict={
                            tf_description_embedding: current_description_embedding,
                            tf_question: current_question,
                            tf_mc_answers: current_mc_answers,
                            tf_label: current_answers
                            })
        else:
            _, loss = sess.run(
                    [train_op, tf_loss],
                    feed_dict={
                        tf_question: current_question,
                        tf_mc_answers: current_mc_answers,
                        tf_label: current_answers
                        })

        current_learning_rate = lr*decay_factor
        lr.assign(current_learning_rate).eval()

        tStop = time.time()
        if np.mod(itr, 10) == 0:
            print ("Iteration: ", itr, " Loss: ", loss, " Learning Rate: ", lr.eval())
            print ("Time Cost:", round(tStop - tStart,2), "s")
        # if np.mod(itr, 15000) == 0:
        if itr in [20, 40, 50, 60, 80, 100, 200, 300, 400, 800, 1200]:
            print ("Iteration ", itr, " is done. Saving the model ...")
            saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step=itr)
            losses.append({'iteration': str(itr), 'loss': str(loss)})  # my code


    json.dump(losses, open(checkpoint_path+'losses.json', 'w'))   # my code
    print ("Finally, saving the model ...")
    saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step=n_epochs)
    tStop_total = time.time()
    print ("Total Time Cost:", round(tStop_total - tStart_total,2), "s")
    sess.close() 
    gc.collect()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--variation', required=True, help='enter variation - isq, iq, sq, q')
    parser.add_argument('--offline_text', required=True, help='enter variation - True, False')
    # parser.add_argument('--checkpoint_path', required=True, help='enter checkpoint_path - model_save/')

    args = parser.parse_args()
    params = vars(args)

    offline_text = params['offline_text']

    if offline_text:
        checkpoint_path = 'model_save_offline/'
    else:
        checkpoint_path = 'model_save/'

    variation = params['variation']
    checkpoint_path = checkpoint_path + 'model_save_' + variation + '/'
    # "model_save/model_save_q/"

    with tf.device('/gpu:'+str(0)):
        train()

