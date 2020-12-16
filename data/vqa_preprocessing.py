'''
Adapted from https://github.com/chingyaoc/VQA-tensorflow/blob/master/data/vqa_preprocessing.py
'''


import json
import os
import argparse


def main(params):
    '''
    Creates raw json file including image urls, story and question along with their corresponding annotations.
    Parameters: 
        params (list of input parameters): input question and annotations file
    Returns:
        None 
    '''

    train = []
    test = []
    # imdir='%s/%s'

    print('Loading annotations and questions...')
    train_anno = json.load(open(params['train_annotations'], 'r', encoding='utf-8'))
    val_anno = json.load(open(params['test_annotations'], 'r', encoding='utf-8'))

    train_ques = json.load(open(params['train_questions'], 'r', encoding='utf-8'))
    val_ques = json.load(open(params['test_questions'], 'r', encoding='utf-8'))

    # directory = ''

    # train instances   
    for i in range(len(train_anno['annotations'])):
        ans = train_anno['annotations'][i]['multiple_choice_answer']
        question_id = train_anno['annotations'][i]['question_id']
        images = train_anno['annotations'][i]['image_id']
        image_path = []
        # changing the format of image urls to match the image file names
        for image in images:
            image = str(image)
            img = image[8:].replace('/', '-')
            image_path = img

        question = train_ques['questions'][i]['question']
        
        # if the variation requires summary in the input 
        if params['variation'] in ['suq', 'isuq']:
            description = train_ques['questions'][i]['summary']
        else:
            description = train_ques['questions'][i]['description']

        train.append({'ques_id': question_id, 'img_path': image_path, 'description': description, 
                      'question': question, 'ans': ans
                      })
        
    # validation instances
    for i in range(len(val_anno['annotations'])):
        ans = val_anno['annotations'][i]['multiple_choice_answer']
        question_id = val_anno['annotations'][i]['question_id']
        images = val_anno['annotations'][i]['image_id']
        image_path = []
        
        # changing the format of image urls to match the image file names
        for image in images:
            image = str(image)
            img = image[8:].replace('/', '-')
            image_path = img
        
        question = val_ques['questions'][i]['question']
        
        # if the variation requires summary in the input 
        if params['variation'] in ['suq', 'isuq']:
            description = val_ques['questions'][i]['summary']
        else:
            description = val_ques['questions'][i]['description']

        test.append({'ques_id': question_id, 'img_path': image_path, 'description': description, 
            'question': question, 'ans': ans 
            })

    print('Training sample %d, Testing sample %d...' %(len(train), len(test)))

    # storing both train and val instances into json files
    json.dump(train, open('vqa_raw_train.json', 'w'))
    json.dump(test, open('vqa_raw_test.json', 'w'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--train_questions', required=True, help='input train questions file')
    parser.add_argument('--test_questions', required=True, help='input test questions file')
    parser.add_argument('--train_annotations', required=True, help='input train annotations file')
    parser.add_argument('--test_annotations', required=True, help='input test annotations file')
    
    # adding variation parameter for pipelined summary questions
    # if summary has to be used then replace description with summary 
    # isq - images, story, question
    # suq - summary, question
    # isuq - images, summary, question
    parser.add_argument('--variation', default='isq', help='variation - isq, suq, isuq')
  
    args = parser.parse_args()
    params = vars(args)
    main(params)
