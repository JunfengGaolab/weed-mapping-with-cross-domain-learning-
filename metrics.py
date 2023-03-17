import _init_paths
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
from pydensecrf.utils import unary_from_labels
import pydensecrf.densecrf as dcrf
from glob import glob
import cv2
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
from DeepLabv3 import relu6, BilinearUpsampling
from Utils import MaxPoolingWithArgmax2D, MaxUnpooling2D
from data_loader import weighted_categorical_crossentropy
from Segnet import SegNet
from UNET import UNET
import ntpath
from keras.models import model_from_json

def mIOU(gt, pred):
    ulabels = np.unique(gt)
    iou = np.zeros(len(ulabels))
    for k, u in enumerate(ulabels):
        inter = (gt == u) & (pred == u)
        union = (gt == u) | (pred == u)
        iou[k] = inter.sum() / union.sum()
    return np.round(iou.mean(), 3)

# TODO change the image size according to your tested dataset
def calculate_iou(model_path, nClasses=3, image_size=(512, 512), image_path='../data', gt_path='../data'):
    """
    calculate the iou of the test dataset
    image_size = (512, 512) by default
    """
    image_files = glob(image_path + '/*.jpg') + glob(image_path + '*.png')
    gt_files = glob(gt_path + '/*.jpg') + glob(gt_path + '/*.png')
    ### todo revise the model name
    model_name = model_path.split('_')[-3]
    print('you are evaluating {} model'.format(model_name))

    weight_class = np.array([1, 2, 1.5])

    if model_name == 'DeepLabv3':
        model = load_model(model_path,
                           custom_objects={'relu6': relu6, 'BilinearUpsampling': BilinearUpsampling},
                           compile=False)
    elif model_name == 'SegNet':
        model = load_model(model_path,
                           custom_objects={'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D,
                                           'MaxUnpooling2D': MaxUnpooling2D,
                                           'loss': weighted_categorical_crossentropy(weight_class)})
    else:
        model = load_model(model_path, compile=False)

        # json_file = open('./save_trained_models/weights/UNET_model.json', 'r')
        # model_json = json_file.read()
        # json_file.close()
        # model = model_from_json(model_json, custom_objects={
        #                            'loss': weighted_categorical_crossentropy(weight_class)})
        # # load weights
        # # model = UNET(3, 512, 512)
        # model.load_weights(model_path)

    # or using model = load_model('./save_trained_models/weed_maize_DeepLabv3_weighted_988.h5',
    #                    custom_objects={'relu6': relu6,
    #                                    'BilinearUpsampling': BilinearUpsampling,
    #                                    'loss': weighted_categorical_crossentropy(weights1)})

    image_files = sorted(image_files)
    gt_files = sorted(gt_files)

    label = np.zeros((len(image_files), np.prod(image_size)), dtype='float32')
    X = np.zeros((len(image_files), image_size[1], image_size[0], 3), dtype='float32')
    mask_category = np.zeros((len(image_files), image_size[1], image_size[0], 3), dtype='float32')

    # print(model.metrics_names)

    for i, (image, gt) in enumerate(zip(image_files, gt_files)):
        # print(image.split('.')[0] == gt.split('.')[0])
        img = cv2.imread(image, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.
        grd_truth = cv2.imread(gt, 1)
        grd_truth = cv2.cvtColor(grd_truth, cv2.COLOR_BGR2RGB)
        grd_truth = cv2.cvtColor(grd_truth, cv2.COLOR_RGB2GRAY)
        grd_cat = np_utils.to_categorical(grd_truth, num_classes=3)
        # print('grd_cat::', grd_cat.shape)
        # labels = np.unique(mask)
        grd_truth = grd_truth.astype('int32')
        y = grd_truth.flatten()
        X[i, :, :, :] = img
        label[i, :] = y
        mask_category[i, :, :, :] = grd_cat

    preds = model.predict(X, batch_size=12) # ignore setting batch_size in this row
    #  todo when setting compile=False, the following codes not working
    # scores = model.evaluate(X, mask_category)
    # print('!'*20)
    # print(scores) # one is loss value, one is accuracy
    # print("%s: %.3f%%" % ("Mean Pixel Accuracy: ", scores[1] * 100))

    conf_m = np.zeros((nClasses, nClasses), dtype=float)
    mask = np.reshape(np.argmax(preds, axis=-1), (-1,) + image_size)
    flat_pred = np.ravel(mask).astype('int')
    flat_gt = np.ravel(label).astype('int')
    conf_m = confusion_matrix(flat_gt, flat_pred)
    # for p, g in zip(flat_pred, flat_gt):
    #     # if g == nClasses:
    #     #     continue
    #     if p < nClasses and g < nClasses:
    #         conf_m[p, g] += 1
    #     else:
    #         print('Invalid entry encountered, skipping!')
    I = np.diag(conf_m)
    U = np.sum(conf_m, axis=0) + np.sum(conf_m, axis=1) - I
    IOU = I / U
    print ('the IOU of soil: , weed: , crop are {}'.format(IOU))
    mean_IOU = np.mean(IOU)
    print('the mean_IOU: {}'.format(mean_IOU))
    return conf_m

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    trained_classes = classes
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=11)
    tick_marks = np.arange(len(classes))
    plt.xticks(np.arange(len(trained_classes)), classes, rotation=45, fontsize=10)
    plt.yticks(tick_marks, classes,fontsize=10)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, np.round(cm[i, j], 3), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=11)
    plt.tight_layout()
    plt.ylabel('True label',fontsize=11)
    plt.xlabel('Predicted label',fontsize=11)
    return cm

# Fully connected CRF post processing function
def do_crf(im, mask, zero_unsure=True):
    colors, labels = np.unique(mask, return_inverse=True)
    print('colour is {}'.format(colors))
    print('label shape is '.format(labels.shape))
    image_size = mask.shape[:2]
    n_labels = len(set(labels.flat))
    d = dcrf.DenseCRF2D(image_size[1], image_size[0], n_labels)  # width, height, nlabels
    U = unary_from_labels(labels, n_labels, gt_prob=.7, zero_unsure=zero_unsure)
    print('U shape is {}'.format(U.shape))
    d.setUnaryEnergy(U)
    # This adds the color-independent term, features are the locations only.
    # d.addPairwiseGaussian(sxy=(3,3), compat=3)
    # # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    # # im is an image-array, e.g. im.dtype == np.uint8 and im.shape == (640,480,3)
    # d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=im.astype('uint8'), compat=10)
    # Q = d.inference(5) # 5 - num of iterations

    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=im.astype('uint8'),
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)
    MAP = np.argmax(Q, axis=0).reshape(image_size)
    unique_map = np.unique(MAP)
    print('MAP shape is {}'.format(MAP.shape))
    print(unique_map)
    for u in unique_map: # get original labels back
        np.putmask(MAP, MAP == u, colors[u])
    return MAP
    # MAP = do_crf(frame, labels.astype('int32'), zero_unsure=False)


if __name__ == "__main__":
    # image_path = '/vsc-mounts/gent-data/423/vsc42313/data/test_real/imgs'
    # gt_path = '/vsc-mounts/gent-data/423/vsc42313/data/test_real/anns'
    image_path = '/vsc-mounts/gent-data/423/vsc42313/data/random_test/UAV-image-test/16x16_resized'
    gt_path = '/vsc-mounts/gent-data/423/vsc42313/data/random_test/UAV-image-test/16x16_resized_GT'

    ## TODO REVISE THE PATH OF THE TRAINED MODELS
    model_path = '/ddn1/vol1/site_scratch/leuven/423/vsc42313/save_trained_models/model_checkpoint_saved/weed_maize_SegNet_5412_weight4.h5'

    conf = calculate_iou(model_path=model_path, image_path=image_path, gt_path=gt_path)
    # print(conf)
    classes = ['soil', 'weed', 'crop']
    plt.figure()
    cm = plot_confusion_matrix(conf, classes=classes)
    print('soil, weed, crop accuracy:{}'.format(np.diag(cm)))
    plt.title('Mean IOU: ' + str(np.round(np.diag(cm).mean(), 3)))
    plt.colorbar()
    plt.tight_layout()
    ## TODO revise the name of saved figure
    plt.savefig('DEEP_{}_weighted3_2985.png'.format(model_path.split('_')[-3]))
    plt.close()












