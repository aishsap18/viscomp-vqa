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

class Answer_Generator():
    def __init__(self, rnn_size, rnn_layer, batch_size, input_embedding_size, dim_image, dim_hidden, max_words_q, vocabulary_size, max_words_d, vocabulary_size_d, drop_out_rate):
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
        self.embed_state_desc_b = tf.Variable(tf.random.uniform([self.dim_hidden], -0.08, 0.08), name='embed_state_desc_b')

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

    	# end my code

        # score-embedding
        self.embed_scor_W = tf.Variable(tf.random.uniform([dim_hidden, num_output], -0.08, 0.08), name='embed_scor_W')
        self.embed_scor_b = tf.Variable(tf.random.uniform([num_output], -0.08, 0.08), name='embed_scor_b')

    def build_model(self):
        image = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.dim_image])
        
        # my code
        image2 = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.dim_image])
        image3 = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.dim_image])
        image4 = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.dim_image])
        image5 = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.dim_image])
        # end my code

        question = tf.compat.v1.placeholder(tf.int32, [self.batch_size, self.max_words_q])
        description = tf.compat.v1.placeholder(tf.int32, [self.batch_size, self.max_words_d])
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
        # print("\nstate0: {} \n state1: {} \n".format(state_drop[0], state_drop[1]))
        state_linear = tf.compat.v1.nn.xw_plus_b(tf.concat([state_drop[0], state_drop[1]], 1), self.embed_state_W, self.embed_state_b)
        state_emb = tf.tanh(state_linear)

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

        # multimodal (fusing question & image)
        # question
        state_drop = tf.nn.dropout(state, 1-self.drop_out_rate)
        # print("\nstate0: {} \n state1: {} \n".format(state_drop[0], state_drop[1]))
        state_linear = tf.compat.v1.nn.xw_plus_b(tf.concat([state_drop[0], state_drop[1]], 1), self.embed_state_W, self.embed_state_b)
        state_emb = tf.tanh(state_linear)

        # description
        state_drop_d = tf.nn.dropout(state_d, 1-self.drop_out_rate)
        # print("\nstate0: {} \n state1: {} \n".format(state_drop[0], state_drop[1]))
        state_linear_d = tf.compat.v1.nn.xw_plus_b(tf.concat([state_drop_d[0], state_drop_d[1]], 1), self.embed_state_desc_W, self.embed_state_desc_b)
        state_emb_d = tf.tanh(state_linear_d)

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
        # my code end

        # scores = tf.multiply(state_emb, image_emb)
        scores = state_emb * state_emb_d * image_emb 
        scores_drop = tf.nn.dropout(scores, 1-self.drop_out_rate)
        scores_emb = tf.compat.v1.nn.xw_plus_b(scores_drop, self.embed_scor_W, self.embed_scor_b) 

    	# Calculate cross entropy
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores_emb, labels=label)

    	# Calculate loss
        loss = tf.reduce_mean(cross_entropy)

        return loss, image, image2, image3, image4, image5, description, question, label   # my code
        # return loss, image, question, label
    
    def build_generator(self):
        image = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.dim_image])
        # my code
        image2 = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.dim_image])
        image3 = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.dim_image])
        image4 = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.dim_image])
        image5 = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.dim_image])
        # my code end

        question = tf.compat.v1.placeholder(tf.int32, [self.batch_size, self.max_words_q])
        description = tf.compat.v1.placeholder(tf.int32, [self.batch_size, self.max_words_d])

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
        # my code end

        # scores = tf.multiply(state_emb, image_emb)
        scores = state_emb * state_emb_d * image_emb 
        scores_drop = tf.nn.dropout(scores, 1-self.drop_out_rate)
        scores_emb = tf.compat.v1.nn.xw_plus_b(scores_drop, self.embed_scor_W, self.embed_scor_b) 

        # FINAL ANSWER
        generated_ANS = tf.compat.v1.nn.xw_plus_b(scores_drop, self.embed_scor_W, self.embed_scor_b)

        return generated_ANS, image, image2, image3, image4, image5, description, question # my code
        # return generated_ANS, image, question
    
#####################################################
#                 Global Parameters		    #  
#####################################################
print('Loading parameters ...')
# Data input setting
input_img_h5 = None 
input_ques_h5 = None 
input_json = None 

# Train Parameters setting
learning_rate = 0.0003			# learning rate for rmsprop
#starter_learning_rate = 3e-4
learning_rate_decay_start = -1		# at what iteration to start decaying learning rate? (-1 = dont)
# batch_size = 500			# batch_size for each iterations
batch_size = 64
input_embedding_size = 200		# he encoding size of each token in the vocabulary
rnn_size = 512				# size of the rnn in number of hidden nodes in each layer
rnn_layer = 2				# number of the rnn layer
dim_image = 4096
dim_hidden = 1024 #1024			# size of the common embedding vector


num_output = 1896			# number of output answers


img_norm = 1				# normalize the image feature. 1 = normalize, 0 = not normalize
decay_factor = 0.99997592083

# Check point
checkpoint_path = None # 'model_save/'


# misc
gpu_id = 0
# max_itr = 150000
max_itr = 3000  # 30000
# n_epochs = 5  # 3000
max_words_q = 15
max_words_d = 50

num_answer = None # 1896
# num_answer = 131



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

    # load image feature
    print('loading image feature...')
    with h5py.File(input_img_h5,'r') as hf:
        # -----0~82459------
        tem = hf.get('images_train')
        img_feature = np.array(tem)
    # load h5 file
    print('loading h5 file...')
    with h5py.File(input_ques_h5,'r') as hf:
        # total number of training data is 215375
        # question is (26, )
        tem = hf.get('ques_train')
        train_data['question'] = np.array(tem)-1
        # max length is 23
        tem = hf.get('ques_length_train')
        train_data['length_q'] = np.array(tem)

        # question is 
        tem = hf.get('description_train')
        train_data['description'] = np.array(tem)-1
        # max length is 
        tem = hf.get('description_length_train')
        train_data['length_d'] = np.array(tem)

        # total 82460 img
        tem = hf.get('img_pos_train')
	    # convert into 0~82459
        train_data['img_list'] = np.array(tem)-1
        # answer is 1~1000
        tem = hf.get('answers')
        train_data['answers'] = np.array(tem)-1

    print('question aligning')
    train_data['question'] = right_align(train_data['question'], train_data['length_q'])

    print('description aligning')
    train_data['description'] = right_align(train_data['description'], train_data['length_d'])

    print('Normalizing image feature')
    if img_norm:
        tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature), axis=1))
        print('tem: {}'.format(tem.shape))
        print('image feature dim: {}'.format(img_feature.shape))
        img_feature = np.divide(img_feature, np.transpose(np.tile(tem,(4096,1))))

    return dataset, img_feature, train_data


def train():
    print ('loading dataset...')
    dataset, img_feature, train_data = get_data()
    num_train = train_data['question'].shape[0]
    vocabulary_size = len(dataset['ix_to_word'].keys())
    vocabulary_size_d = len(dataset['ix_to_word_d'].keys()) 
    print ('vocabulary_size : ' + str(vocabulary_size))
    print ('vocabulary_size_d : ' + str(vocabulary_size_d))

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
            drop_out_rate = 0.5)

    # tf_loss, tf_image, tf_question, tf_label = model.build_model()
    tf_loss, tf_image, tf_image2, tf_image3, tf_image4, tf_image5, tf_description, tf_question, tf_label = model.build_model() # my code

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

        current_description = train_data['description'][index,:]
        current_length_d = train_data['length_d'][index]

        current_answers = train_data['answers'][index]
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
        # my code end

        # do the training process!!!
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
                    tf_label: current_answers
                    })

        current_learning_rate = lr*decay_factor
        lr.assign(current_learning_rate).eval()

        tStop = time.time()
        if np.mod(itr, 100) == 0:
            print ("Iteration: ", itr, " Loss: ", loss, " Learning Rate: ", lr.eval())
            print ("Time Cost:", round(tStop - tStart,2), "s")
        if np.mod(itr, 500) == 0:
            print ("Iteration ", itr, " is done. Saving the model ...")
            saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step=itr)
            losses.append({'iteration': str(itr), 'loss': str(loss)})  # my code


    json.dump(losses, open('losses.json', 'w'))   # my code
    print ("Finally, saving the model ...")
    saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step=itr)
    tStop_total = time.time()
    print ("Total Time Cost:", round(tStop_total - tStart_total,2), "s")
    sess.close() 
    gc.collect()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_img_h5', required=True, help='enter input image features path')
    parser.add_argument('--input_data_h5', required=True, help='enter input data h5 path')
    parser.add_argument('--input_data_json', required=True, help='enter input data json path')
    parser.add_argument('--num_ans', required=True, help='enter number of answers')
    parser.add_argument('--model_path', required=True, help='enter checkpoint_path - model_save/')

    args = parser.parse_args()
    params = vars(args)

    input_img_h5 = params['input_img_h5']
    input_ques_h5 = params['input_data_h5']
    input_json = params['input_data_json']
    num_answer = params['num_ans']
    checkpoint_path = params['model_path']

    with tf.device('/gpu:'+str(0)):
        train()

