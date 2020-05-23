import sys
import os.path
import argparse
import numpy as np
import h5py
import json

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

def load_data(imgs):

    inputs = []
    targets = []

    for img in imgs:
        if variation in ['sq', 'isq', 'bsq', 'bisq']:
            input_s = img['description'] + ' ' + img['question']
        elif variation in ['iq', 'q', 'bq', 'biq']:
            input_s = img['question']

        target_s = img['ans']
        inputs.append(input_s)
        targets.append(target_s)

    unique_img, img_pos = get_unqiue_img(imgs)

    return inputs, targets, img_pos, unique_img

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--train_input', required=True, help='enter train input json')
    parser.add_argument('--test_input', required=True, help='enter test input json')
    parser.add_argument('--output_json', required=True, help='enter output json')
    parser.add_argument('--variation', required=True, help='enter variation - isq, iq, sq, q, bsq, bq, bisq, biq')

    args = parser.parse_args()
    params = vars(args)

    # variation = sq, q (b - bert embeddings)
    variation = params['variation']

    imgs_train = json.load(open(params['train_input'], 'r'))
    imgs_test = json.load(open(params['test_input'], 'r'))
    
    out = {}

    inputs, targets, img_pos, unique_img = load_data(imgs_train)
    inputs_test, targets_test, img_pos_test, unique_img_test = load_data(imgs_test)
    
    out['img_pos'] = img_pos.tolist()
    out['unique_img_train'] = unique_img

    out['img_pos_test'] = img_pos_test.tolist()
    out['unique_img_test'] = unique_img_test        
 
    out['inputs'] = inputs
    out['targets'] = targets

    out['inputs_test'] = inputs_test
    out['targets_test'] = targets_test

    json.dump(out, open(params['output_json'], 'w'))
    print('wrote ', params['output_json'])
