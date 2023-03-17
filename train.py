import _init_paths
import numpy as np
from FCN32 import FCN32
from Segnet import SegNet
from FCN8 import FCN8
from UNET import UNET
from UNET_no_VGG import unet
from DeepLabv3 import Deeplabv3
from Seg_U_Net import Seg_UNet
from data_loader import train_generator, weighted_categorical_crossentropy
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import *
from keras.layers import *
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *
from keras.utils.np_utils import to_categorical
from keras.utils import multi_gpu_model
from Utils import MaxUnpooling2D, MaxPoolingWithArgmax2D



# class_weight = np.array([1, 4, 3.18]) # this class weights too large for training,
# over predict weed pixels
# class_weight = np.array([1, 1.5, 1.2])
# class_weight = np.array([1, 2.5, 1.5]) # conventional used weighted without considering UAV image
# class_weight = np.array([1, 8.5, 3.5])
# class_weight = np.array([1, 3.5, 2])
class_weight = np.array([1, 8, 8])
custom_loss = weighted_categorical_crossentropy(class_weight)

# model = vgg10_unet(input_shape=(512, 512, 3), weights='imagenet')
# model = FCN32(3, 512, 512)
# model = FCN32(3, 512, 512)
# model = SegNet(3, 512, 512)
key = 'SegNet'
method = {"FCN32_activation": FCN32, "FCN8": FCN8, 'SegNet': SegNet,
          'UNET': UNET, 'UNET_no_VGG': unet, 'DeepLabv3': Deeplabv3, 'Seg_UNet': Seg_UNet}
model = method[key](3, 512, 512)

### using multiple gpus for training

##########  !!!!!!!!!!!!! continue train the model !!!!!!!!!!!!!!!!   ##############
# continued_model = load_model('./save_trained_models/model_check_point_saved/weed_maize_SegNet2_2765.h5',
#                        custom_objects={'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D,
#                                        'MaxUnpooling2D': MaxUnpooling2D,
#                                        'loss': weighted_categorical_crossentropy(class_weight)})

print(model.summary())
# refer to https://keras.io/losses/
save_model_path = '/ddn1/vol1/site_scratch/leuven/423/vsc42313/'

# model_json = model.to_json()
# with open("./save_trained_models/weights/{}_model.json".format(key), 'w') as json_file:
#     json_file.write(model_json)

model_name = save_model_path + 'save_trained_models/model_checkpoint_saved/weed_maize_{}_layer_5412_weight6.h5'.format(key)

# TODO change the save directory if it is needed
model_checkpoint = ModelCheckpoint(model_name,
                                   monitor='loss', verbose=1,
                                   save_best_only=True)
parallel_model = multi_gpu_model(model, gpus=4)
# for index in range(15):
#     model.layers[index].trainable = True
parallel_model.compile(optimizer=Adam(lr=1e-3), loss=custom_loss, metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=1,
                              min_delta=0.0001, cooldown=0, min_lr=0)

# model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
# change model to parallel_model if it is applied
# loss = ''



batch_size = 24  # gpu=3 set batch size 18, gpu=1, set batch size 6
# TODO change the steps_per_epoch based on real number of training samples
history = parallel_model.fit_generator(train_generator(batch_size=batch_size),
                                       steps_per_epoch=5412 // batch_size,
                                       epochs=400,
                                       # validation_data=train_generator(batch_size=2),
                                       # validation_steps=1,
                                       callbacks=[model_checkpoint, reduce_lr])


## change the model to be parallel_model if it is applied here

##########  !!!!!!!!!!!!! continue train the model !!!!!!!!!!!!!!!!   ##############
# new_model = load_model('../Segmentation/save_trained_models/model_check_point_saved/weed_maize_SegNet2_2765.h5',
#                        custom_objects={'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D,
#                                        'MaxUnpooling2D': MaxUnpooling2D,
#                                        'loss': weighted_categorical_crossentropy(class_weight)})