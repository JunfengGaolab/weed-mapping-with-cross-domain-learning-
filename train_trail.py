# import _init_paths
# from FCN32 import FCN32
# from Segnet import SegNet
# from FCN8 import FCN8
# from UNET import UNET
# from UNET_no_VGG import unet
# from DeepLabv3 import Deeplabv3
# from Seg_U_Net import Seg_UNet
# # from data_loader import train_generator, weighted_categorical_crossentropy
# from data_loader_test import weighted_categorical_crossentropy, SegmentationGenerator
# from keras.callbacks import ModelCheckpoint
# from keras.models import *
# from keras.layers import *
# import tensorflow as tf
# from keras.applications.vgg16 import VGG16
# from keras.preprocessing.image import ImageDataGenerator
# from keras.optimizers import *
# from keras.utils.np_utils import to_categorical
# from keras.utils import multi_gpu_model
# from keras.callbacks import ReduceLROnPlateau
# from keras.models import model_from_json
#
# # model = vgg10_unet(input_shape=(512, 512, 3), weights='imagenet')
# # model = FCN32(3, 512, 512)
# # model = FCN32(3, 512, 512)
# # model = SegNet(3, 512, 512)
# key = 'SegNet'
# method = {"FCN32_activation": FCN32, "FCN8": FCN8, 'SegNet': SegNet,
#           'UNET': UNET, 'UNET_no_VGG': unet, 'DeepLabv3': Deeplabv3,'Seg_UNet': Seg_UNet}
# model = method[key](3, 512, 512)
#
# ## class weights
# class_weights = np.array([1, 3.5, 2])
# custom_loss = weighted_categorical_crossentropy(class_weights)
#
# ### using multiple gpus for training
# parallel_model = multi_gpu_model(model, gpus=3)
# # for index in range(15):
# #     model.layers[index].trainable = True
# parallel_model.compile(optimizer=Adam(lr=1e-4), loss=custom_loss, metrics=['accuracy'])
# # model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
# # change model to parallel_model if it is applied
# # loss = ''
# print('inputs: ', [input.op.name for input in model.inputs])
#
# print('outputs: ', [output.op.name for output in model.outputs])
# print(model.summary())
# # refer to https://keras.io/losses/
# # model_checkpoint = ModelCheckpoint('./save_trained_models/unet_weed_maize_{}.h5'.format(key), monitor='loss', verbose=1,
# #                                    save_best_only=True, save_weights_only=True)
# batch_size = 18 # gpu=3 set batch size 18, gpu=1, set batch size 6
# train_generator = SegmentationGenerator(mode='train', n_classes=3, batch_size=batch_size, validation_split=0.1,
#                  crop_shape=(512, 512), resize_shape=None, seed=7, horizontal_flip=True, blur=0,
#                  vertical_flip=0, brightness=True, rotation=5.0, zoom=0.1, do_ahisteq=False)
#
# valid_generator = SegmentationGenerator(mode='validation', n_classes=3, batch_size=batch_size, validation_split=0.1,
#                  crop_shape=(512, 512), resize_shape=None, seed=7, horizontal_flip=True, blur=0,
#                  vertical_flip=0, brightness=True, rotation=5.0, zoom=0.1, do_ahisteq=False)
# # train_generator = train_generator(batch_size=batch_size)
# history = parallel_model.fit_generator(train_generator,
#                                        steps_per_epoch= 2489 // batch_size,
#                                        epochs=400,
#                                        validation_data=valid_generator,
#                                        validation_steps=276 // batch_size,
#                                        )
# model.save_weights('./save_trained_models/weights/weed_maize_{}_weighted3_test_2765.h5'.format(key))
# ## change the model to be parallel_model if it is applied here
#


import _init_paths
from FCN32 import FCN32
from Segnet import SegNet
from Seg_U_Net import Seg_UNet
from FCN8 import FCN8
from UNET import UNET
from DeepLabv3 import Deeplabv3
from UNET_no_VGG import unet
from data_loader import train_generator, weighted_categorical_crossentropy
from keras.callbacks import ModelCheckpoint
from keras.models import *
from keras.layers import *
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *
from keras.utils.np_utils import to_categorical
from keras.utils import multi_gpu_model

# model = vgg10_unet(input_shape=(512, 512, 3), weights='imagenet')
# model = FCN32(3, 512, 512)
# model = FCN32(3, 512, 512)
# model = SegNet(3, 512, 512)
key = 'SegUNet'
method = {"FCN32_activation": FCN32, "FCN8": FCN8, 'SegNet': SegNet,
          'UNET': UNET, 'UNET_no_VGG': unet, 'DeepLabv3': Deeplabv3, 'SegUNet': Seg_UNet}
model = method[key](3, 512, 512)

class_weight = np.array([1, 8, 8])
custom_loss = weighted_categorical_crossentropy(class_weight)


### using multiple gpus for training
parallel_model = multi_gpu_model(model, gpus=4)
# for index in range(15):
#     model.layers[index].trainable = True
parallel_model.compile(optimizer=Adam(lr=1e-3), loss=custom_loss, metrics=['accuracy'])
# model.compile(optimizer=Adam(lr=3e-4), loss='categorical_crossentropy', metrics=['accuracy'])
# learning rate 3e-4
# change model to parallel_model if it is applied
# loss = ''
print(model.summary())
# refer to https://keras.io/losses/
# model_checkpoint = ModelCheckpoint('./save_trained_models/unet_weed_maize_{}.h5'.format(key), monitor='loss', verbose=1,
#                                    save_best_only=True, save_weights_only=True)
# TODO setting the batch_size based on your own compution resource and
# TODO setting the number of steps_per_epoch based on your own training dataset
batch_size = 24 # gpu=3 set batch size 18, gpu=1, set batch size 6
loss = parallel_model.fit_generator(train_generator(batch_size=batch_size),
                                       steps_per_epoch= 5412 // batch_size,
                                       epochs= 300)

save_model_path = '/ddn1/vol1/site_scratch/leuven/423/vsc42313/'
model_name = save_model_path + 'save_trained_models/model_saved/weed_maize_layer_{}_5412_weight6.h5'.format(key)
model.save(model_name)
## change the model to be parallel_model if it is applied here