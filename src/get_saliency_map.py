import os
import numpy as np
import model as m
import cv2
from PIL import Image
import argparse
import tensorflow as tf
import keras
import sys
sys.path.append('../utils')
#from read_data import Loader
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import array_to_img
from keras.models import Model
from keras.layers import Input


root_dict = {'pascal': '/lfs1/data/PASCAL-S/datasets/imgs/pascal',
             'flms': '/lfs1/data/FLMS/image',
             'cuhk': '/lfs1/users/hzhang/project/crop/data/cuhk_img',
             'fcd': '/lfs1/data/FCD/image',
             'ava': '/lfs1/users/hzhang/project/crop/data/ava_imgs/ava_hq'}
def get_shape(img, ratio, resize):
    w, h = img.size
    size = (w, h)
    # print size, args.ratio, args.resize
    if ratio == 1:
        if resize is not None:
            size = (resize, resize)
    else:
        if resize is not None:
            if w <= h:
                size = (resize, resize * h / w)
            else:
                size = (resize * w / h, resize)
    image = img.resize(size, Image.ANTIALIAS)
    return image


def test(model, dataset, ratio, resize):

    save_path = 'saliency_result_05'
    save_path = os.path.join(save_path, dataset)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    if dataset in root_dict:
        data_root = root_dict[dataset]
    else:
        data_root = dataset
    #data_root = '/lfs1/data/CUHK-ICD/images'
    #data_root = '/lfs1/data/FCD/image'
    datas = os.listdir(data_root)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    for image_pair in datas:

        name = image_pair.split('.')[0]
        #print name
        if not image_pair.endswith('.jpg'):
            continue
        if dataset != 'ava':
            img = Image.open(os.path.join(data_root, image_pair))
        else:
            img = Image.open(os.path.join('/lfs1/data/AVA-dataset/image', image_pair))
        img = img.convert('RGB')
        img = get_shape(img, ratio, resize)

        image = np.asarray(img)

        h1, w1 = image.shape[0], image.shape[1]

        h2, w2 = (image.shape[0] / 16 + 1) * 16, (image.shape[1] / 16 + 1) * 16
        image = cv2.copyMakeBorder(image, top=0, bottom=h2 - h1, left=0, right=w2 - w1, borderType=cv2.BORDER_CONSTANT,
                                   value=0)
        image = image.astype('float32') / 255.0
        image = np.reshape(image, (1, image.shape[0], image.shape[1], 3))

        saliency_map = model.predict(image, batch_size=1, verbose=0)
        saliency_map = saliency_map.reshape(h2, w2, 1)
        label_map = (saliency_map > 0.5).astype('float32')
        label_map *= 255.0
        #saliency_map = saliency_map * 255.0
        #saliency_map = (saliency_map > 30.0)
        #out_image = array_to_img(saliency_map)
        out_image = array_to_img(label_map)
        out_image = out_image.resize((w1, h1), Image.ANTIALIAS)
        if dataset == 'pascal':
            out_image.save(os.path.join(save_path, name + '.png'))
        else:
            out_image.save(os.path.join(save_path, name + '.jpg'))
    return save_path

def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    set_session(tf.Session(config=config))

    model = m.SaliencyUnet(state='test').BuildModel()
    weight_base = 'models'
    weight_file = os.path.join(weight_base, 'saliency_05.h5')
    model.load_weights(weight_file)
    test(model, sys.argv[1], None, None)

if __name__ == '__main__':
    main()
