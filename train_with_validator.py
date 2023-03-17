import _init_paths
from FCN32 import FCN32
from Segnet import SegNet
from FCN8 import FCN8
from UNET import UNET
from UNET_no_VGG import unet
from DeepLabv3 import Deeplabv3
from Seg_U_Net import Seg_UNet
# from data_loader import train_generator, weighted_categorical_crossentropy
from data_loader_test import weighted_categorical_crossentropy, SegmentationGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import *
from keras.layers import *
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *
from keras.utils.np_utils import to_categorical
from keras.utils import multi_gpu_model
from keras.callbacks import ReduceLROnPlateau
from keras.models import model_from_json

# model = vgg10_unet(input_shape=(512, 512, 3), weights='imagenet')
# TODO CHANGE THE weights for each class if the number of training samples imbalance
# class_weights = np.array([1, 2, 1.5])
# class_weights = np.array([1, 1.5, 1.5])
class_weights = np.array([1, 2.5, 1.5])
# hyperparameter need to be tuned, original value [1, 55, 24], log value [1, 4, 3.18]
# class_weights = np.array([1, 50, 24])
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


custom_loss = weighted_categorical_crossentropy(class_weights)
key = 'Seg_UNet'
method = {"FCN32": FCN32, "FCN8": FCN8, 'SegNet': SegNet,
          'UNET': UNET, 'UNET_no_VGG': unet, 'DeepLabv3': Deeplabv3, 'Seg_UNet': Seg_UNet}

model = method[key](3, 512, 512)

print(model.summary())
# refer to https://keras.io/losses/
## save model as json file
#### TODO CHANGE THE DIRECTORY OF SAVE PATH ####
save_model_path = '/ddn1/vol1/site_scratch/leuven/423/vsc42313/'

model_json = model.to_json()
# with open("./save_trained_models/weights/{}_model.json".format(key), 'w') as json_file:
#     json_file.write(model_json)

# TODO change the save directory if it is needed
model_name = save_model_path + 'save_trained_models/model_checkpoint_saved/weed_maize_{}_.h5'.format(key)
# model_name = './save_trained_models/model_checkpoint_saved/weed_maize_{}_synthetic1.h5'.format(key)
model_checkpoint = ModelCheckpoint(model_name, monitor='val_loss',
                                   verbose=1, save_best_only=True, save_weights_only=False)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1,
                              min_delta=0.0001, cooldown=0, min_lr=0)

batch_size = 25  # gpu=3 set batch size 18, gpu=1, set batch size 6

### using multiple gpus for training
# for index in range(15):
#     model.layers[index].trainable = True
model = multi_gpu_model(model, gpus=3)
model.compile(optimizer=Adam(lr=0.004), loss=custom_loss, metrics=['accuracy'])
# model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
# change model to parallel_model if it is applied
# loss = ''
# TODO change the steps_per_epoch based on real number of training samples
train_generator = SegmentationGenerator(mode='train', n_classes=3, batch_size=batch_size, validation_split=0.1,
                 crop_shape=(512, 512), resize_shape=None, seed=7, horizontal_flip=True, blur=3,
                 vertical_flip=True, brightness=True, rotation=5.0, zoom=0.2, do_ahisteq=False)

valid_generator = SegmentationGenerator(mode='validation', n_classes=3, batch_size=batch_size, validation_split=0.1,
                 crop_shape=(512, 512), resize_shape=None, seed=7, horizontal_flip=True, blur=0,
                 vertical_flip=0, brightness=True, rotation=5.0, zoom=0.1, do_ahisteq=False)
# train_generator = train_generator(batch_size=batch_size)
history = model.fit_generator(train_generator,
                                       steps_per_epoch = 5722 // batch_size,
                                       epochs=300,
                                       validation_data = valid_generator,
                                       validation_steps=635 // batch_size,
                                       callbacks=[model_checkpoint, reduce_lr])

## change the model to be parallel_model if it is applied here