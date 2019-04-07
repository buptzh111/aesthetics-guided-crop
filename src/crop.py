try:
    import keras
    import tensorflow as tf
    import cv2
    import sys
    import os
    import numpy as np
    import time
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

    #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    img_path = sys.argv[1]
    if os.path.isfile(img_path):
        img_names = [img_path]
    elif os.path.isdir(img_path):
        img_names = os.listdir(img_path)
        img_names = [os.path.join(img_path, img_name) for img_name in img_names]
    else:
        raise Exception("The input path is not a image file or a image directory.")

    crop_img_save_path = sys.argv[2] if sys.argv[2] is not None else 'crop_result'
    if not os.path.isdir(crop_img_save_path):
        os.makedirs(crop_img_save_path)

    #set model
    model_saliency = m.SaliencyUnet(state='test').BuildModel()
    model_regression = m.ofn_net(state='test').set_model()
    model_saliency.load_weights('models/saliency.h5')
    model_regression.load_weights('models/regression.h5')

    crop_regions = []
    saliency_regions = []

    for i in img_names:
        if not i.endswith(('.jpg', '.jpeg', '.png')):
            continue
        image_name = i.split('/')[-1]

        image = cv2.imread(i, -1)
        h, w = image.shape[0], image.shape[1]

        h2, w2 = (h / 16 + 1) * 16, (w / 16 + 1) * 16
        img = cv2.copyMakeBorder(image, top=0, bottom=h2 - h, left=0, right=w2 - w, borderType=cv2.BORDER_CONSTANT,
                                 value=0)
        print image.shape, img.shape
        exit()
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        saliency_map = model_saliency.predict(img, batch_size=1, verbose=0)
        saliency_map = saliency_map.reshape(h2, w2, 1)
        saliency_image = cv2.resize(saliency_map, (224, 224), interpolation=cv2.INTER_CUBIC)

        saliency_box = Minimum_Rectangle(saliency_image, r=0.9)

        saliency_box = [saliency_box[0], saliency_box[0] + saliency_box[3],
                        saliency_box[1], saliency_box[1] + saliency_box[2]]
        saliency_box = normalization(224, 224, saliency_box)
        saliency_region = recover_from_normalization_with_order(w - 1, h - 1, saliency_box)
        saliency_img = image[saliency_region[1]: saliency_region[3],saliency_region[0]: saliency_region[2],:]
        w3, h3 = saliency_region[2] - saliency_region[0], saliency_region[3] - saliency_region[1]

        if w3 <= h3:
            saliency_img = cv2.resize(saliency_img, (224, h3 * 224 / w3), interpolation=cv2.INTER_CUBIC)
        else:
            saliency_img = cv2.resize(saliency_img, (w3 * 224 / h3, 224), interpolation=cv2.INTER_CUBIC)
        saliency_image = np.expand_dims(saliency_img, axis=0).astype('float32')
        saliency_image /= 255.0

        offset = model_regression.predict(saliency_image, batch_size=1)[0]
        final_region = add_offset(w, h, saliency_box, offset)

        final_region = recover_from_normalization_with_order(w, h, final_region)

        top_left_final = tuple(final_region[:2])
        down_right_final = tuple(final_region[2:])
        top_left_saliency = tuple(saliency_region[:2])
        down_right_saliency = tuple(saliency_region[2:])
        # draw crop box on original image.

        cv2.rectangle(image, top_left_saliency, down_right_saliency, (0, 255, 0), 2)
        cv2.rectangle(image, top_left_final, down_right_final, (0, 0, 255), 2)

        cv2.imwrite(os.path.join(crop_img_save_path, image_name), image)
        # save crop box as txt file.
        final_region_to_file = ' '.join([image_name] + [str(u) for u in final_region])
        saliency_region_to_file = ' '.join([image_name] + [str(u) for u in saliency_region])
        crop_regions.append(final_region_to_file)
        saliency_regions.append(saliency_region_to_file)
    
    with open(os.path.join(crop_img_save_path, 'crop_region_coordinate.txt'), 'w') as f:
        f.write('\n'.join(crop_regions))
    with open(os.path.join(crop_img_save_path, 'saliency_region_coordinate.txt'), 'w') as f:
        f.write('\n'.join(saliency_regions))

if __name__ == '__main__':
    crop()
