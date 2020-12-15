from model_viscomp_vqa import *
import tensorflow as tf
import numpy as np
import os, h5py, sys, argparse
import time
import math
import json
from tensorflow.python.client import device_lib


input_json = None
input_img_h5 = None
input_ques_h5 = None
input_text_h5 = None


def get_data_test():
    dataset = {}
    test_data = {}
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
            tem = hf.get('images_test')
            img_feature = np.array(tem)
    
    # load h5 file
    print('loading h5 file...')
    with h5py.File(input_ques_h5,'r') as hf:
        test_data['num_answers'] = np.array(hf.get('num_answers'))
        print("num_answers: {}".format(test_data['num_answers']))
        # total number of training data is 215375
        # question is (26, )
        tem = hf.get('ques_test')
        test_data['question'] = np.array(tem)-1
        # max length is 23
        tem = hf.get('ques_length_test')
        test_data['length_q'] = np.array(tem)

        if variation in ['isq', 'sq']:
            if offline_text == "False":
                # print("\n\n\noffline_text false\n\n\n")
                # question is 
                tem = hf.get('description_test')
                test_data['description'] = np.array(tem)-1
                # max length is 
                tem = hf.get('description_length_test')
                test_data['length_d'] = np.array(tem)
            else:
                # print("\n\n\noffline_text true\n\n\n")
                with h5py.File(input_text_h5,'r') as hf_text:
                    tem = hf_text.get('text_test')
                    test_data['description_embedding'] = np.array(tem)

        if variation in ['isq', 'iq']:
            # total 82460 img
            tem = hf.get('img_pos_test')
            # convert into 0~82459
            test_data['img_list'] = np.array(tem)-1
        
        # quiestion id
        tem = hf.get('question_id_test')
        test_data['ques_id'] = np.array(tem)
        
        tem = hf.get('MC_ans_test')
        test_data['MC_ans_test'] = np.array(tem)

    print('question aligning')
    test_data['question'] = right_align(test_data['question'], test_data['length_q'])

    if variation == 'q':
        return dataset, test_data

    if variation in ['isq', 'sq']:
        if offline_text == "False":
            print('description aligning')
            test_data['description'] = right_align(test_data['description'], test_data['length_d'])

        if variation == 'sq':
            return dataset, test_data

    if variation in ['isq', 'iq']:
        print('Normalizing image feature')
        if img_norm:
            tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature), axis=1))
            img_feature = np.divide(img_feature, np.transpose(np.tile(tem,(4096,1))))

        return dataset, img_feature, test_data


def softmax(generated_ans):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(generated_ans) / np.sum(np.exp(generated_ans), axis=0)


def test(model_path, result_number, result_path):
    # print('\n\nvariation: {}\n\n'.format(variation))
    print ('loading dataset...')
    if variation in ['isq', 'iq']:
        dataset, img_feature, test_data = get_data_test()
    else:
        dataset, test_data = get_data_test()

    num_test = test_data['question'].shape[0]
    vocabulary_size = len(dataset['ix_to_word'].keys())
    print ('vocabulary_size : ' + str(vocabulary_size))

    if variation in ['isq', 'sq']:
        vocabulary_size_d = len(dataset['ix_to_word_d'].keys()) 
        print ('vocabulary_size_d : ' + str(vocabulary_size_d))
    else:
        vocabulary_size_d = 0

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
            drop_out_rate = 0,
            num_answers = test_data['num_answers'],
            variation=variation,
            offline_text=offline_text)

    # tf_answer, tf_image, tf_image2, tf_image3, tf_image4, tf_image5, tf_description, tf_question, tf_mc_answers = model.build_generator() # my code
    # tf_answer, tf_image, tf_question, = model.build_generator()
    if variation == 'isq':
        if offline_text == "False":
            # print("\n\n\noffline_text false\n\n\n")
            tf_answer, tf_image, tf_image2, tf_image3, tf_image4, tf_image5, tf_description, tf_question, tf_mc_answers = model.build_generator() # my code
        else:
            # print("\n\n\noffline_text true\n\n\n")
            tf_answer, tf_image, tf_image2, tf_image3, tf_image4, tf_image5, tf_description_embedding, tf_question, tf_mc_answers = model.build_generator() # my code
    elif variation == 'iq':
        tf_answer, tf_image, tf_image2, tf_image3, tf_image4, tf_image5, tf_question, tf_mc_answers = model.build_generator() # my code
    elif variation == 'sq':
        if offline_text == "False":
            tf_answer, tf_description, tf_question, tf_mc_answers = model.build_generator() # my code
        else:
            tf_answer, tf_description_embedding, tf_question, tf_mc_answers = model.build_generator() # my code
    else:
        tf_answer, tf_question, tf_mc_answers = model.build_generator() # my code

    #sess = tf.InteractiveSession()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    tStart_total = time.time()
    result = []
    for current_batch_start_idx in range(0,num_test,batch_size):
    #for current_batch_start_idx in xrange(0,3,batch_size):
        tStart = time.time()
        # set data into current*
        if current_batch_start_idx + batch_size < num_test:
            current_batch_file_idx = range(current_batch_start_idx,current_batch_start_idx+batch_size)
        else:
            current_batch_file_idx = range(current_batch_start_idx,num_test)

        current_question = test_data['question'][current_batch_file_idx,:]
        current_length_q = test_data['length_q'][current_batch_file_idx]

        if variation in ['isq', 'sq']:
            if offline_text == "False":
                # print("\n\n\noffline_text false\n\n\n")
                current_description = test_data['description'][current_batch_file_idx,:]
                current_length_d = test_data['length_d'][current_batch_file_idx]
            else:
                # print("\n\n\noffline_text true\n\n\n")
                current_description_embedding = test_data['description_embedding'][current_batch_file_idx,:]
                # print("current_description_embedding: {}".format(current_description_embedding))

        current_ques_id  = test_data['ques_id'][current_batch_file_idx]

        if variation in ['isq', 'iq']:
            current_img_list = test_data['img_list'][current_batch_file_idx][:,0]
            current_img = img_feature[current_img_list,:] # (batch_size, dim_image)
            # my code
            current_img2_list = test_data['img_list'][current_batch_file_idx][:,1]
            current_img2 = img_feature[current_img2_list,:]
            current_img3_list = test_data['img_list'][current_batch_file_idx][:,2]
            current_img3 = img_feature[current_img3_list,:]
            current_img4_list = test_data['img_list'][current_batch_file_idx][:,3]
            current_img4 = img_feature[current_img4_list,:]
            current_img5_list = test_data['img_list'][current_batch_file_idx][:,4]
            current_img5 = img_feature[current_img5_list,:]

        current_mc_answers = test_data['MC_ans_test'][current_batch_file_idx]
        # my code end

        # deal with the last batch
        # if(len(current_img)<500):
        if(len(current_question)<batch_size):
            
            pad_q = np.zeros((batch_size-len(current_question),max_words_q),dtype=np.int)
            pad_q_len = np.zeros(batch_size-len(current_length_q),dtype=np.int)
            
            if variation in ['isq', 'sq']:
                if offline_text == "False":
                    # print("\n\n\noffline_text false\n\n\n")
                    pad_d = np.zeros((batch_size-len(current_question),max_words_d),dtype=np.int)
                    pad_d_len = np.zeros(batch_size-len(current_length_d),dtype=np.int)
                    current_description = np.concatenate((current_description, pad_d))
                    current_length_d = np.concatenate((current_length_d, pad_d_len))
                else:
                    # print("\n\n\noffline_text true\n\n\n")
                    pad_d_emb = np.zeros((batch_size-len(current_description_embedding),dim_hidden),dtype=np.float)
                    current_description_embedding = np.concatenate((current_description_embedding, pad_d_emb))
                    # print("padded current_description_embedding: {}".format(current_description_embedding))
            
            pad_q_id = np.zeros(batch_size-len(current_length_q),dtype=np.int)
            pad_ques_id = np.zeros(batch_size-len(current_length_q),dtype=np.int)
            
            pad_img_list = np.zeros(batch_size-len(current_length_q),dtype=np.int)
            
            current_question = np.concatenate((current_question, pad_q))
            current_length_q = np.concatenate((current_length_q, pad_q_len))
            
            current_ques_id = np.concatenate((current_ques_id, pad_q_id))
            
            if variation in ['isq', 'iq']:
                pad_img = np.zeros((batch_size-len(current_img),dim_image),dtype=np.int)
                current_img = np.concatenate((current_img, pad_img))
                current_img_list = np.concatenate((current_img_list, pad_img_list))
                # my code
                current_img2 = np.concatenate((current_img2, pad_img))
                current_img2_list = np.concatenate((current_img2_list, pad_img_list)) 
                current_img3 = np.concatenate((current_img3, pad_img))
                current_img3_list = np.concatenate((current_img3_list, pad_img_list)) 
                current_img4 = np.concatenate((current_img4, pad_img))
                current_img4_list = np.concatenate((current_img4_list, pad_img_list)) 
                current_img5 = np.concatenate((current_img5, pad_img))
                current_img5_list = np.concatenate((current_img5_list, pad_img_list)) 

            pad_mc = np.zeros((batch_size-len(current_mc_answers),5),dtype=np.int)
            current_mc_answers = np.concatenate((current_mc_answers, pad_mc))           
            # my code end

        if variation == 'isq':
            if offline_text == "False":
                # print("\n\n\noffline_text false\n\n\n")
                generated_ans = sess.run(
                                    tf_answer,
                                    feed_dict={
                                        tf_image: current_img,
                                        tf_image2: current_img2,    # my code
                                        tf_image3: current_img3,    # my code
                                        tf_image4: current_img4,    # my code
                                        tf_image5: current_img5,    # my code
                                        tf_description: current_description,
                                        tf_question: current_question,
                                        tf_mc_answers: current_mc_answers
                                        })
            else:
                # print("\n\n\noffline_text true\n\n\n")
                generated_ans = sess.run(
                                    tf_answer,
                                    feed_dict={
                                        tf_image: current_img,
                                        tf_image2: current_img2,    # my code
                                        tf_image3: current_img3,    # my code
                                        tf_image4: current_img4,    # my code
                                        tf_image5: current_img5,    # my code
                                        tf_description_embedding: current_description_embedding,
                                        tf_question: current_question,
                                        tf_mc_answers: current_mc_answers
                                        })
        elif variation == 'iq':
            generated_ans = sess.run(
                                tf_answer,
                                feed_dict={
                                    tf_image: current_img,
                                    tf_image2: current_img2,    # my code
                                    tf_image3: current_img3,    # my code
                                    tf_image4: current_img4,    # my code
                                    tf_image5: current_img5,    # my code
                                    tf_question: current_question,
                                    tf_mc_answers: current_mc_answers
                                    })
        elif variation == 'sq':
            if offline_text == "False":
                generated_ans = sess.run(
                                    tf_answer,
                                    feed_dict={
                                        tf_description: current_description,
                                        tf_question: current_question,
                                        tf_mc_answers: current_mc_answers
                                        })
            else:
                generated_ans = sess.run(
                                    tf_answer,
                                    feed_dict={
                                        tf_description_embedding: current_description_embedding,
                                        tf_question: current_question,
                                        tf_mc_answers: current_mc_answers
                                        })
        else:
            generated_ans = sess.run(
                                tf_answer,
                                feed_dict={
                                    tf_question: current_question,
                                    tf_mc_answers: current_mc_answers
                                    })


        # initialize json list
        for i in range(0,batch_size):
            prob_dist = softmax(generated_ans[i])
            
            if current_batch_start_idx != 0:
                if(current_ques_id[i] == 0):
                    continue
            
            options = dataset['test_mc_ans_ix'][str(current_batch_file_idx[i]+1)]
            true_ans = options[dataset['test_ans_ix'][current_batch_file_idx[i]]]

            
            final_ans = np.argmax(prob_dist)
            ans = options[final_ans]

            result.append({u'question_id': str(current_ques_id[i]), 
                u'true_answer': 'unk' if true_ans == '0' else dataset['ix_to_ans'][true_ans],
                u'predicted_answer': 'unk' if ans == '0' else dataset['ix_to_ans'][ans],
                u'multiple_choices': [dataset['ix_to_ans'][opt] if opt != '0' else 'unk' for opt in options]
                })

        tStop = time.time()
        print ("Testing batch: ", current_batch_file_idx[0])
        print ("Time Cost:", round(tStop - tStart,2), "s")


    print ("Testing done.")
    tStop_total = time.time()
    print ("Total Time Cost:", round(tStop_total - tStart_total,2), "s")
    # Save to JSON
    print ('Saving result...')
    my_list = list(result)
    dd = json.dump(my_list,open(result_path+'/result_'+result_number+'.json','w'))

variation = ''
offline_text = None

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--variation', required=True, help='enter variation - isq, iq, sq, q')
    parser.add_argument('--input_img_h5', required=True, help='enter input image features path')
    parser.add_argument('--input_data_h5', required=True, help='enter input data h5 path')
    parser.add_argument('--input_data_json', required=True, help='enter input data json path')
    parser.add_argument('--offline_text', required=True, help='enter variation - True/False if True enter path')
    parser.add_argument('--input_text_h5', default=None, help='enter input text h5 path')
    parser.add_argument('--model_path', required=True, help='input model name to be used')
    parser.add_argument('--result_path', required=True, help='path to store result')
    
    args = parser.parse_args()
    params = vars(args)

    offline_text = params['offline_text']
    variation = params['variation']
    input_json = params['input_data_json']
    input_img_h5 = params['input_img_h5']
    input_ques_h5 = params['input_data_h5']
    input_text_h5 = params['input_text_h5']
    model_path = params['model_path']
    result_path = params['result_path']
    result_number = model_path.split('-')[-1]
    # "model_save/model_save_q/model-10"

    # print("hello")
    gpu = [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"]

    # print(gpu)
    with tf.device('/gpu:'+str(0)):
        test(model_path, result_number, result_path)
