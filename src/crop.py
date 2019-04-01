try:
    import keras
    import tensorflow as tf
    import cv2
    import sys
    import os
    import numpy as np
except:
    print 'Lack of corresponding dependencies.'

import keras.backend as K
if K.backend() != 'tensorflow':
    raise Exception("Only 'tensorflow' is supported as backend")

from utils import *
from keras.preprocessing.image import img_to_array, array_to_img
from PIL import Image, ImageDraw
import model as m

def crop():

    img_path = sys.argv[1]
    if os.path.isfile(img_path):
        img_names = [img_path]
    elif os.path.isdir(img_path):
        img_names = os.listdir(img_path)
        img_names = [os.path.join(img_path, img_name) for img_name in img_names]
    else:
        raise Exception("The input path is not a image file or a image directory.")

    crop_img_save_path = 'crop_result'
    if not os.path.isdir(crop_img_save_path):
        os.makedirs(crop_img_save_path)

    #set model
    model_saliency = m.SaliencyUnet(state='test').BuildModel()
    model_regression = m.ofn_net(state='test').set_model()
    model_saliency.load_weights('model/saliency.h5')
    model_regression.load_weights('model/regression.h5')

    crop_regions = []
    saliency_regions = []
    for i in img_names:

        if not i.endswith(('.jpg', '.jpeg', '.png')):
            continue
        image_name = i.split('/')[-1]
        image = Image.open(i)
        img = image.convert('RGB')

        img = np.asarray(img)
        image_for_draw = img.copy()
        h1, w1 = img.shape[0], img.shape[1]

        h2, w2 = (h1 / 16 + 1) * 16, (w1 / 16 + 1) * 16
        img = cv2.copyMakeBorder(img, top=0, bottom=h2 - h1, left=0, right=w2 - w1, borderType=cv2.BORDER_CONSTANT,
                                   value=0)
        img = img.astype('float32') / 255.0
        img = np.reshape(img, (1, h2, w2, 3))
        # z = np.zeros((1, 1, 4))

        # offset = model1.predict(image, batch_size=1, verbose=0)[0]
        saliency_map = model_saliency.predict(img, batch_size=1, verbose=0)
        saliency_map = saliency_map.reshape(h2, w2, 1)
        saliency_map = saliency_map * 255.0
        saliency_image = array_to_img(saliency_map)
        saliency_image = saliency_image.resize((224, 224), Image.ANTIALIAS)
        saliency_map = np.asarray(saliency_image)
        saliency_map = saliency_map.astype('float32') / 255.0
        #print saliency_map.shape
        #saliency_map = np.squeeze(saliency_map, axis=-1)
        saliency_box = Minimum_Rectangle(saliency_map, r=0.9)
        #'''
        saliency_box = [saliency_box[0], saliency_box[0] + saliency_box[3],
                        saliency_box[1], saliency_box[1] + saliency_box[2]]
        saliency_box = normalization(224, 224, saliency_box)
        saliency_region = recover_from_normalization_with_order(w1 - 1, h1 - 1, saliency_box)
        saliency_img = image.crop(saliency_region)
        w3, h3 = saliency_img.size

        if w3 <= h3:
            saliency_img = saliency_img.resize((224, h3 * 224 / w3), Image.ANTIALIAS)
        else:
            saliency_img = saliency_img.resize((w3 * 224 / h3, 224), Image.ANTIALIAS)
        saliency_image = img_to_array(saliency_img)
        saliency_image = np.expand_dims(saliency_image, axis=0)
        saliency_image /= 255.0
        offset = model_regression.predict(saliency_image, batch_size=1)[0]
        final_region = add_offset(w1, h1, saliency_box, offset)
        #print saliency_box, final_region
        #exit()
        #print saliency_box
        final_region = recover_from_normalization_with_order(w1, h1, final_region)

        #top_left_final = tuple(final_region[:2])
        #down_right_final = tuple(final_region[2:])
        #top_left_saliency = tuple(saliency_region[:2])
        #down_right_saliency = tuple(saliency_region[2:])
        # draw crop box on original image.
        pr = ImageDraw.Draw(image)
        pr.rectangle(saliency_region, None, 'blue')
        pr.rectangle(final_region, None, 'yellow')
        image.save(os.path.join(crop_img_save_path, image_name))
        # save crop box as txt file.
        final_region_to_file = ' '.join([image_name] + [str(u) for u in final_region])
        saliency_region_to_file = ' '.join([image_name] + [str(u) for u in saliency_region])
        crop_regions.append(final_region_to_file)
        saliency_regions.append(saliency_region_to_file)
    with open(os.path.join(crop_img_save_path, 'crop_region_coordinate.txt'), 'w') as f:
        f.write('\n'.join(final_regions))
    with open(os.path.join(crop_img_save_path, 'saliency_region_coordinate.txt'), 'w') as f:
        f.write('\n'.join(saliency_regions))

if __name__ == '__main__':
    crop()

