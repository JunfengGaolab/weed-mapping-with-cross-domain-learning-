import _init_paths
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np
from DeepLabv3 import relu6, BilinearUpsampling
from Utils import MaxUnpooling2D, MaxPoolingWithArgmax2D
from UNET_no_VGG import unet
from FCN8 import FCN8
from keras import models
from data_loader import weighted_categorical_crossentropy
from keras.layers import Conv2D
from keras import layers
import keras
from keras.applications import VGG16
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
class_weight = np.array([1, 3.5, 2])
key = 'SegNet'
# key = 'DeepLabv3'
# method = {'SegNet': ['SegNet'], 'FCN32': FCN32, 'FCN8': FCN8, 'UNET': UNET, 'Deeplabv3+': Deeplabv3}
# model = unet(3, 512, 512)
# model.load_weights('./save_trained_models/unet_weed_maize_{}_para.h5'.format(key))
# model = load_model('./save_trained_models/unet_weed_maize_UNET.h5')
# model = load_model('./save_trained_models/unet_weed_maize_{}_test.h5'.format(key),
#                    custom_objects={'relu6': relu6, 'BilinearUpsampling': BilinearUpsampling})

# model = load_model('G:\\deep_segmentation\\Segmentation\\save_trained_models\\weed_maize_{}_5412_weight5.h5'.format(key),
#                    custom_objects={'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D,
#                                        'MaxUnpooling2D': MaxUnpooling2D,
#                                    'loss': weighted_categorical_crossentropy(class_weight)})
model = load_model('G:\\deep_segmentation\\Segmentation\\save_trained_models\\unet_weed_maize_FCN32_activation_test.h5')
model.summary()


# img_path = '../data/test/imgs/_DSC3391_0.jpg'
img_path = 'E:\\PhD_annotated\\field_image_and_annotation\\split_image2\\_DSC8676_2.jpg'
img = image.load_img(img_path, target_size=(512, 512))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, 0)
img_tensor = img_tensor/127.5 - 1.
print(img_tensor.shape)

layer_outputs = [layer.output for layer in model.layers[1:]] # TODO:  model.layers[1:4]] start with 1 as layer 0 is input images
# Creates a model that will return these outputs, given the model input:
print(layer_outputs)
visualization_model = models.Model(inputs=model.input, outputs=layer_outputs)

successive_feature_maps = visualization_model.predict(img_tensor)
# first_layer_activation = activations[0]
# print(first_layer_activation.shape)
# plt.imshow(first_layer_activation[0, :, :, 2], cmap='bwr')
# plt.show()


# plot and stack all the channels side by side
# read the layer names of the model

# layer_names = [layer.name for layer in model.layers]
# for layer_name, feature_map in zip(layer_names, successive_feature_maps):
#     print(feature_map.shape)
#     if len(feature_map.shape) == 4:
#         # plot the feature map for the conv / maxpool layer not the fully connected layers
#         n_features = feature_map.shape[-1]
#         size = feature_map.shape[1] # feature map shape (1, size, size, n_feature)
#         # we will tile our images in this matrix
#         display_grid = np.zeros((size, size*n_features))
#         for i in range (n_features):
#             x = feature_map[0,:,:,i]
#             x -= x.mean()
#             x /= x.std()
#             x *= 64
#             x += 128
#             x = np.clip(x, 0, 255).astype('uint8')
#             display_grid[:, i*size : (i+1)*size] = x # tile each filter into a horizaontal grid
#         # ----
#         # display the grid
#         # ----
#         scale = 20. / n_features
#         plt.figure(figsize=(scale*n_features, scale))
#         plt.title(layer_name)
#         plt.grid(False)
# plt.imshow(display_grid, aspect='auto', cmap='jet')
# plt.show()



####
## save this piece of codes for


layer_names = []
for layer in model.layers: # model.layers[1:4]
    layer_names.append(layer.name)
image_per_row = 16
# display our feature map
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    # this is number of features in the feature map
    if len(feature_map.shape) == 4:
        n_features = feature_map.shape[-1]
        # the feature map has the shape (1, size, size, n_features)
        size = feature_map.shape[1]

        # tile the activation channels in the matrix
        n_cols = n_features // image_per_row
        display_grid = np.zeros((size*n_cols, image_per_row*size))
        for col in range(n_cols):
            for row in range(image_per_row):
                channel_image = feature_map[0, :, :, col * image_per_row + row]
                channel_image = np.squeeze(channel_image)
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        # plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.imshow(display_grid / 255.0, aspect='auto', cmap='jet')
        plt.colorbar()
        # plt.savefig('feature_map_{}.jpg'.format(layer_name), dpi=600)
        plt.show()

###############
## save this piece of codes above


############# TODO visualize the convert filters ###################
# # model = unet(3, 512, 512)
# # layer_name = 'conv2d_3'
# layer_name = 'expanded_conv_1_expand'
# filter_index = 0
# layer_output = model.get_layer(layer_name).output
# loss = K.mean(layer_output[:, :, :, filter_index])
# print(loss.shape)
# # the call to 'gradients' returns a list of tensors (of size 1 in this case)
# # hence we only keep the first element -- which is a tensor
# grads = K.gradients(loss, model.input)[0]
# # A non-obvious trick to use for the gradient descent process to go smoothly is to normalize the gradient tensor,
# #  by dividing it by its L2 norm (the square root of the average of the square of the values in the tensor).
# #  This ensures that the magnitude of the updates done to the input image is always within a same range
#
# # we add the 1e-5 before dividing so as to avoid accidentally dividing by 0
# grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
# iterate = K.function([model.input], [loss, grads])
# # testing
# loss_value, grads_value = iterate([np.zeros((1, 512, 512, 3))])
#
# # we start from a gray image with some noise
# input_img_data = np.random.random((1, 50, 50,3))*20 + 128
# # run gradient ascent for 40 steos
# step = 1.
# for i in range(40):
#     # compute the loss value and gradient value
#     loss_value, grads_value = iterate([input_img_data])
#     # here we just adjust the input image in the direction that maximize the loss
#     input_img_data += grads_value * step
#
#
# def process_image(x):
#     # normalize tensor: center on 0., ensure std is 0.1
#     x -= x.mean()
#     x /= (x.std() + 1e-5)
#     x *= 0.1
#     # clip to [0., 1]
#     x += 0.5
#     x = np.clip(x, 0, 1)
#     # oonvert to RGB array
#     x *= 255
#     x = np.clip(x, 0, 255).astype('uint8')
#     return x
#
# def generate_pattern(layer_name, filter_index=0, size=512):
#     # build a loss function that maximum the activetion
#     layer_output = model.get_layer(layer_name).output
#     loss = K.mean(layer_output[:, :, :, filter_index])
#
#     # compute the gradient of the input picture with this loss
#     grads = K.gradients(loss, model.input)[0]
#     grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
#     # This function returns the loss and grads given the input picture
#     iterate = K.function([model.input], [loss, grads])
#
#     # input_img_data = np.random.random((1, size, size, 3)) * 20 + 128
#     input_img_data = img_tensor
#     # run gradient ascent for 40 steps
#     step = 1.
#     for i in range(40):
#         loss_value, grads_value = iterate([input_img_data])
#         input_img_data += grads_value * step
#     img = input_img_data[0]
#     return process_image(img)
# plt.imshow(generate_pattern('expanded_conv_1_expand', 9))
# plt.show()



