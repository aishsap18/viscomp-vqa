'''
Adapted from https://github.com/TingAnChien/san-vqa-tensorflow/blob/master/prepro_img.py
'''


import numpy as np
import cv2
import os, pdb
import caffe
import h5py, json
from skimage.transform import resize
from progress.bar import Bar
import argparse

[IMG_HEIGHT, IMG_WIDTH] = [224, 224]

input_json = None

image_root = None 
# vgg19
cnn_proto = None 
cnn_model = None 
gpuid = 0
batch_size = 10
out_name = None


def extract_feat(imlist, dname):
    '''
    Extracting and storing VGG 19 (fc7 layer) image features.
    Parameters:
        imlist (list of str): list of image ids
        dname (str): train/val images
    Returns:
        None
    '''

    print("\ndata length: {}".format(len(imlist)))
    dataLen = len(imlist)
    bar = Bar('Processing {}'.format(dname), max=dataLen/batch_size+1)
    # vgg19
    # f.create_dataset(dname, (dataLen, 512, 7, 7), dtype='f4') # pool5
    f.create_dataset(dname, (dataLen, 4096), dtype='f4') # fc7
    batch = list(zip(range(0, dataLen, batch_size), range(batch_size, dataLen+1, batch_size)))    
    batch.append(((dataLen//batch_size)*batch_size, dataLen))
    
    for start, end in batch:
        batch_image = np.zeros([batch_size, 3, IMG_HEIGHT, IMG_WIDTH])
        batch_imname = imlist[start:end]
        for b in range(end-start):
            # use batch_imname[b].encode('utf-8') if error
            imname = os.path.join(image_root, batch_imname[b].split('/')[-1])
            print("Image: {}".format(imname))
            # load the image and resize it
            I = resize(cv2.imread(imname), (IMG_HEIGHT, IMG_WIDTH))-mean
            I = np.transpose(I, (2, 0, 1))
            batch_image[b, ...] = I
         
        net.blobs['data'].data[:] = batch_image
        net.forward()
        # extract vgg19 features 
        # batch_feat = net.blobs['pool5'].data[...].copy()
        batch_feat = net.blobs['fc7'].data[...].copy()
        # store in the dataset
        f[dname][start:end] = batch_feat[:end-start, ...]
        bar.next()
    bar.finish()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # inputs
    parser.add_argument('--input_json', required=True, help='enter json input')
    parser.add_argument('--image_root', required=True, help='enter root directory of images')
    parser.add_argument('--cnn_proto', required=True, help='enter cnn prototxt')
    parser.add_argument('--cnn_model', required=True, help='enter cnn model')
    parser.add_argument('--out_name', default='data_img.h5', help='enter output h5 file')

    args = parser.parse_args()
    params = vars(args)

    input_json = params['input_json']
    image_root = params['image_root']
    cnn_proto = params['cnn_proto']
    cnn_model = params['cnn_model']
    out_name = params['out_name']

    caffe.set_device(gpuid)
    caffe.set_mode_gpu()

    # load the caffe model
    net = caffe.Net(cnn_proto, caffe.TEST, weights=cnn_model)

    mean = np.array((103.939, 116.779, 123.680), dtype=np.float32)

    # load the input data
    with open(input_json) as data_file:
        data = json.load(data_file)

    f = h5py.File(out_name, "w")

    # extract features of train/val images
    extract_feat(data['unique_img_train'], 'images_train')
    extract_feat(data['unique_img_test'], 'images_test')

    f.close()

