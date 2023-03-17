import os
import cv2
import random
import tensorflow as tf
from keras.utils.data_utils import Sequence
import numpy as np
from keras.models import *
from keras.layers import *
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.utils import multi_gpu_model
from glob import glob
from sklearn.utils import class_weight
import keras.backend as K
from skimage import exposure
import skimage.transform



def transform(img, mask, blur=3, num_classes=3, brightness=True, rotation=True):
    """

    this function used for data augumentation during training

    blur:   should be odd and set to be 0 to turn off blur
    :param img: images for training
    :param mask: masks (using preprocessing to convert the mask image, all the values are 0, 1, 2)
    :return: coverted_img and converted_mask

    notes: if opencv not working, the using functions from skimage

    """
    # converted_img = [] #np.zeros(img1.shape)
    # converted_mask = []#np.zeros(img1.shape)
    # print(type(converted_img))
    # for i in range(num):
    # img = img1[i, :, :, :]
    # mask = mask1[i, :, :, 0]
    # print(mask.shape) (512, 512)
    # mask = np.expand_dims(mask, axis=2)
    # now the shape of mask is (512, 512, 1)
    mask = np.squeeze(mask)
    if brightness and random.randint(0, 1):
        factor = 1 + random.uniform(0, 1)
        if random.randint(0, 1):
            factor = 1. / factor
        img = exposure.adjust_gamma(img, factor)
        img = cv2.convertScaleAbs(img, alpha=factor, beta=50)
        # table = np.array([((i / 255.)**factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        # img = cv2.LUT(img, table)

    # if blur and random.randint(0, 1):
    intensity = np.random.choice([2*i+1 for i in range(15)])
    img = cv2.GaussianBlur(img, (intensity, intensity), 0)

    if rotation and random.randint(0, 1) and random.randint(0, 1):
        # img = skimage.transform.rotate(img, 270)
        # mask = skimage.transform.rotate(mask, 270)
        ## the above codes do probably not work well
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # img = cv2.transpose(img)
        # mask = cv2.transpose(mask)
        # img = cv2.flip(img, 0)
        # mask = cv2.flip(mask, 0)
    # converted_img[i, :, :, :] = img / 255.
    # converted_img.append(img/255.)
    # converted_img = np.array(converted_img)
    mask = to_categorical(mask, num_classes=num_classes)
    # converted_mask.append(mask)
    # converted_mask = np.array(converted_mask)
    # converted_mask[i, :, :, :] = mask

    return img, mask



def train_generator(batch_size=2):
    data_gen_args = dict(
                         horizontal_flip=True,
                         vertical_flip=True,
                         zoom_range=0.2
                         )
    ### not use the different data_gen_args for image generator and mask generator
    # data_gen_args_mask = dict(
    #     horizontal_flip=True,
    #     vertical_flip=True,
    #     zoom_range=0.2
    # )

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    seed = 1
    ## TODO change the directory accordingly
    image_generator = image_datagen.flow_from_directory(
        '/ddn1/vol1/site_scratch/leuven/423/vsc42313/data/train_real4/',
        classes=['imgs'],
        class_mode=None,
        batch_size=batch_size,
        color_mode='rgb',
        target_size=(512, 512),
        seed=seed
    )
    ### TODO CHANGE it accordingly
    mask_generator = mask_datagen.flow_from_directory(
        '/ddn1/vol1/site_scratch/leuven/423/vsc42313/data/train_real4/',
        classes=['anns'],
        class_mode=None,
        color_mode='grayscale',
        batch_size=batch_size,
        target_size=(512, 512),
        seed=seed
    )
    # train_generator = zip(image_generator, mask_generator)
    # for (img,mask) in train_generator:
    #     img,mask = adjustdata(img,mask,flag_multi_class = True,num_class = 3)
    #     yield (img,mask)
    # while True:
    #     image_n = image_generator.next()
    #     mask_n = mask_generator.next()
    #     X, Y = [], []
    #     for img, mask in zip(image_n, mask_n):



    while True:
        image_n = image_generator.next()
        mask_n = mask_generator.next()
        #  dimension of image_n : [batch_size, 512, 512, 3]
        #  dimension of mask_n : [batch_size, 512, 512, 1]
        X, Y = [], []
        for (img_i, mask_i) in zip(image_n, mask_n):
            # img = np.array(img).astype('uint8')

            img, mask = transform(img_i, mask_i)
            # print(type(img), img.shape)
            # print(type(mask), mask.shape)
            # img = img / 255.
            # mask = to_categorical(mask, num_classes=3)

            # print(mask.shape)
            # print(img.shape)
            # img = np.expand_dims(img,0)
            # mask = np.expand_dims(mask,0)
            X.append(img/255.)
            Y.append(mask)
        batch_img = np.array(X)
        batch_mask = np.array(Y)
        yield (batch_img, batch_mask)

            # print(np.shape(img))

        # print(np.unique(Y, axis = 0))
def weighted_categorical_crossentropy(weights):
    """ weighted_categorical_crossentropy

        Args:
            * weights: numpy array of shape (C, ) where C is the number of classes
        Returns:
            * weighted categorical crossentropy function
        Usage：
            * weights = np.array([1, 55, 24]) # class one at 1, class 2 55 times of class 1
            loss = weighted_categorical_crossentropy(weights)
            model.compile(loss=loss, optimizer='adam')
    """
    if isinstance(weights,list) or isinstance(weights, np.ndarray):
        weights = K.variable(weights)

    def loss(target, output, from_logits=False):
        if not from_logits:
            output /= tf.reduce_sum(output,
                                    len(output.get_shape()) - 1,
                                    True)
            _epsilon = tf.convert_to_tensor(K.epsilon(), dtype=output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
            weighted_losses = target * tf.log(output) * weights
            return - tf.reduce_sum(weighted_losses,len(output.get_shape()) - 1)
        else:
            raise ValueError('WeightedCategoricalCrossentropy: not valid with logits')
    return loss
