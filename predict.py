import _init_paths
import numpy as np
import os
import cv2
from keras.models import load_model
from glob import glob
from draw import drawRec
from Utils import MaxPoolingWithArgmax2D, MaxUnpooling2D
from DeepLabv3 import relu6, BilinearUpsampling
from Segnet import SegNet
from keras.models import model_from_json
from metrics import do_crf


def dcrf_result_map_image(prediction_dcrf):
    img = np.zeros((prediction_crf.shape[0], prediction_crf.shape[1], 3), dtype=np.uint8)## change the spatial resolution based on your own dataset
    # for np.where calculation
    soil = (prediction_crf == 0)
    weed = (prediction_crf == 1)
    maize = (prediction_crf == 2)
    # np.where(condition[x,y])--> condition true assign x, else y
    img[:, :, 0] = np.where(soil, 0, 0)
    img[:, :, 1] = np.where(maize, 255, 0)
    img[:, :, 2] = np.where(weed, 255, 0)
    return img

def result_map_img(prediction):
    """
    map prediction result to mask image
    :param prediction:
    :return:
    """
    img = np.zeros((512, 512, 3), dtype=np.uint8) ## change the spatial resolution based on your own dataset
    prediction = np.squeeze(prediction)
    argmax_idx = np.argmax(prediction, axis=2)
    # for np.where calculation
    soil = (argmax_idx == 0)
    weed = (argmax_idx == 1)
    maize = (argmax_idx == 2)
# np.where(condition[x,y])--> condition true assign x, else y
    img[:, :, 0] = np.where(soil, 0, 0)
    img[:, :, 1] = np.where(maize, 255, 0)
    img[:, :, 2] = np.where(weed, 255, 0)
    return img


if __name__ == '__main__':

    # using json network structure and weights to load model
    # json_file = open('./SegNet_model.json', 'r')
    # model_json = json_file.read()
    # json_file.close()
    # model = model_from_json(model_json)
    # # load weights
    # model.load_weights('./save_trained_models/weights/weed_maize_SegNet_weighted3_2765.h5')

    # model = load_model('../Segmentation/save_trained_models/model_check_point_saved/weed_maize_UNET_no_VGG_5412.h5',
    #                    compile=False)  # Only for without custom layers

    # when model have custom layers or custom loss function, add the custom_objects in the load_model function
    # model = load_model('./save_trained_models/model_checkpoint_saved/weed_maize_SegNet_synthetic1.h5',
    #                    custom_objects={'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D,
    #                                    'MaxUnpooling2D': MaxUnpooling2D}, compile=False)
    do_crf_signal = False
    print(".........running the model......")
    # model = load_model(
    #     '/ddn1/vol1/site_scratch/leuven/423/vsc42313/save_trained_models/model_checkpoint_saved/weed_maize_SegNet_weighted2_988.h5',
    #                    custom_objects={'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D,
    #                                    'MaxUnpooling2D': MaxUnpooling2D}, compile=False)
    model = load_model(
        '/ddn1/vol1/site_scratch/leuven/423/vsc42313/save_trained_models/model_checkpoint_saved/unet_weed_maize_SegNet.h5',
                       custom_objects={'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D,
                                       'MaxUnpooling2D': MaxUnpooling2D}, compile=False)
    # model = load_model('../Segmentation/save_trained_models/model_check_point_saved/weed_maize_DeepLabv3_5412.h5',
    #                    custom_objects={'relu6': relu6, 'BilinearUpsampling': BilinearUpsampling}, compile=False)
    print(model.summary())

    # model.get_weights('./save_trained_models/unet_weed_maize_Segnet.h5')
    # Segmentation_image_files = glob('/vsc-mounts/gent-data/423/vsc42313/data/random_test/UAV-image-test/16x16_resized/*')
    #                            # + glob('../data/test/imgs/sow thistle/*.png')
    Segmentation_image_files = glob('/vsc-mounts/gent-data/423/vsc42313/data/SLU/raw_images/*')
                               # + glob('../data/test/imgs/sow thistle/*.png')
    Segmentation_image_files.sort() # using sort and zip can draw pair image of prediction and its ground truth
    for img in Segmentation_image_files:
        image = cv2.imread(img)
        img_height, img_width = image.shape[:2]
        image_resized0 = cv2.resize(image, (512, 512))
        image_resized0 = cv2.cvtColor(image_resized0, cv2.COLOR_BGR2RGB)
        image_resized0 = image_resized0 / 255.
        image_resized0 = np.asarray(image_resized0)
        image_resized = np.expand_dims(image_resized0, axis=0)
        prediction = model.predict(image_resized)
        ## prediction shape is (1, 512, 512, 3)
        print(prediction.shape)
        prediction_mask = result_map_img(prediction)

        prediction1 = np.squeeze(prediction)
        prediction_argmax = np.argmax(prediction1, axis=2)

        print('prediction_argmax shape: '.format(prediction_argmax.shape))
        # prediction_crf = do_crf(image_resized0*255, prediction_argmax, zero_unsure=False)
        # print ('prediction_crf shape '.format(prediction_crf.shape))
        # the shape of prediction_crf is (512, 512)
        # prediction_mask1 = dcrf_result_map_image(prediction_crf)

        # prediction_mask_origin = cv2.resize(prediction_mask1, (img_width, img_height),
        #                                     interpolation=cv2.INTER_NEAREST)
        prediction_argmax_resize = cv2.resize(prediction_argmax, (img_width, img_height),
                                            interpolation=cv2.INTER_NEAREST)
        # print('argmax_resize shape {}'.format(prediction_argmax_resize.shape))

        img_name = img.split('/')[-1].split('.')[0]
        # prediction_folder = '../data/test_prediction_result/'
        # prediction_folder = '/vsc-mounts/gent-data/423/vsc42313/data/random_test_result/UAV-test-result/'
        # prediction_addweight_folder = '/vsc-mounts/gent-data/423/vsc42313/data/random_test_result_addweight/UAV-test-addweight/'
        prediction_folder = '/vsc-mounts/gent-data/423/vsc42313/data/SLU/prediction/'
        prediction_addweight_folder = '/vsc-mounts/gent-data/423/vsc42313/data/SLU/prediction-addweight/'
        if not os.path.exists(prediction_folder):
            os.mkdir(prediction_folder)
        if not os.path.exists(prediction_addweight_folder):
            os.mkdir(prediction_addweight_folder)
        # path = '../' + img.split('.')[-2] + '_predict.jpg'
        saved_path = prediction_folder + img_name + '_predict.jpg'
        # print(path)
        save_path_addweight = prediction_addweight_folder + img_name + '_addweight.jpg'



        if do_crf_signal == True:

            prediction_crf = do_crf(image, prediction_argmax_resize, zero_unsure=False)
            print(prediction_crf.shape)
            prediction_mask1 = dcrf_result_map_image(prediction_crf)

            # prediction_mask_origin = cv2.resize(prediction_mask, (img_width, img_height),
            #                                     interpolation=cv2.INTER_NEAREST)
            prediction_addweight = cv2.addWeighted(image, 1, prediction_mask1, 0.7, 0)

            cv2.imwrite(save_path_addweight, prediction_addweight)
            cv2.imwrite(saved_path, prediction_mask1)

        else:
            prediction_mask_origin = cv2.resize(prediction_mask, (img_width, img_height),
                                                interpolation=cv2.INTER_NEAREST)
            prediction_addweight = cv2.addWeighted(image, 1, prediction_mask_origin, 0.7, 0)

            cv2.imwrite(saved_path, prediction_mask_origin)
            cv2.imwrite(save_path_addweight, prediction_addweight)




            # cv2.imwrite(saved_path, prediction_mask_origin)






######    another way to visual prediction result  for more than 3 classes segmentation
# import numpy as np
# import keras
# from PIL import Image
#
# from model import SegNet
#
# import dataset
#
# height = 360
# width = 480
# classes = 12
# epochs = 100
# batch_size = 1
# log_filepath='./logs_100/'
#
# data_shape = 360*480
#
# def writeImage(image, filename):
#     """ label data to colored image """
#     Sky = [128,128,128]
#     Building = [128,0,0]
#     Pole = [192,192,128]
#     Road_marking = [255,69,0]
#     Road = [128,64,128]
#     Pavement = [60,40,222]
#     Tree = [128,128,0]
#     SignSymbol = [192,128,128]
#     Fence = [64,64,128]
#     Car = [64,0,128]
#     Pedestrian = [64,64,0]
#     Bicyclist = [0,128,192]
#     Unlabelled = [0,0,0]
#     r = image.copy()
#     g = image.copy()
#     b = image.copy()
#     label_colours = np.array([Sky, Building, Pole, Road_marking, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
#     for l in range(0,12):
#         r[image==l] = label_colours[l,0]
#         g[image==l] = label_colours[l,1]
#         b[image==l] = label_colours[l,2]
#     rgb = np.zeros((image.shape[0], image.shape[1], 3))
#     rgb[:,:,0] = r/1.0
#     rgb[:,:,1] = g/1.0
#     rgb[:,:,2] = b/1.0
#     im = Image.fromarray(np.uint8(rgb))
#     im.save(filename)
#
# def predict(test):
#     model = keras.models.load_model('seg_100.h5')
#     probs = model.predict(test, batch_size=1)
#
#     prob = probs[0].reshape((height, width, classes)).argmax(axis=2)
#     return prob
#
# def main():
#     print("loading data...")
#     ds = dataset.Dataset(test_file='val.txt', classes=classes)
#     test_X, test_y = ds.load_data('test') # need to implement, y shape is (None, 360, 480, classes)
#     test_X = ds.preprocess_inputs(test_X)
#     test_Y = ds.reshape_labels(test_y)
#
#     prob = predict(test_X)
#     writeImage(prob, 'val.png')
#
# if __name__ == '__main__':
#     main()