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
from data_loader_test import weighted_categorical_crossentropy
import time

weight_class = np.array([1, 1.5, 1.5])
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

    # model = load_model('../Segmentation/save_trained_models/model_check_point_saved/weed_maize_UNET_no_VGG_5412.h5',
    #                    compile=False)  # Only for without custom layers
    start_time = time.time()

    # when model have custom layers or custom loss function, add the custom_objects in the load_model function
    # model = load_model('../Segmentation/save_trained_models/model_check_point_saved/weed_maize_SegNet_200_5412.h5',
    #                    custom_objects={'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D,
    #                                    'MaxUnpooling2D': MaxUnpooling2D}, compile=False)
                                       # 'loss': weighted_categorical_crossentropy(weight_class)}, compile=False)

    model = load_model('../Segmentation/save_trained_models/model_check_point_saved/weed_maize_DeepLabv3_5412.h5',
                       custom_objects={'relu6': relu6, 'BilinearUpsampling': BilinearUpsampling}, compile=False)


    # model.get_weights('./save_trained_models/unet_weed_maize_Segnet.h5')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # TODO change the
    out = cv2.VideoWriter('result_UNet_5412.avi', fourcc, 30, (1080*3, 1920))
    video_path = '../data/mvi_0051.mp4'

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        is_cap_open, frame = cap.read()
        # TODO change the dimension of the input video
        # counter clock wise rotation 90 degree
        frame = cv2.transpose(frame)
        frame = cv2.flip(frame, 0)

        frame_height, frame_width = (1920, 1080)
        if is_cap_open == True:
            fixed_image = cv2.resize(frame, (512, 512))
            # prediction with the trained model
            image_resized = cv2.cvtColor(fixed_image, cv2.COLOR_BGR2RGB)
            image_resized = image_resized / 255.
            image_resized = np.asarray(image_resized)
            image_resized = np.expand_dims(image_resized, axis=0)
            prediction = model.predict(image_resized)
            #  convert to the clored mask
            prediction_mask = result_map_img(prediction)
            prediction_mask_origin = cv2.resize(prediction_mask,
                                                (frame_width, frame_height),
                                                interpolation=cv2.INTER_NEAREST)

        #     show the prediction
        #     result_image = formShowImg(frame, prediction_mask_origin)
            bbox_image = drawRec(frame, prediction_mask_origin)
            print(frame.shape, bbox_image.shape, prediction_mask_origin.shape)
            result_image = np.concatenate((frame, prediction_mask_origin, bbox_image), axis=1)
            # result_image = np.vstack((frame, bbox_image, prediction_mask_origin))
            ## can swtich to use np.hstack(frame, predicton, bbox_image)

            out.write(result_image)
        else:
            break
        # cv2.imshow('predict result', result_image)
        # cv2.waitKey(50)
    cap.release()
    out.release()
    run_time = time.time() - start_time
    print('running time is {} seconds'.format(run_time))
    print('video capture close')













