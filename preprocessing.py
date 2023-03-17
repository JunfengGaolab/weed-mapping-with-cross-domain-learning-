import cv2
import glob
import os
import numpy as np
path = '/vsc-mounts/gent-data/423/vsc42313/data/random_test/UAV-image-test/16x16_resized_GT'
files = os.listdir(path)

##########################  convert clourful mask into categorical RGB mask  ###############################
## this translation works well for dataloader model to_categorical function  ##
i = 0
for img in files:
    x = cv2.imread(os.path.join(path,img))
    gray_x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    # gray_x[np.where(gray_x == 0)] = 0 # background = 0
    gray_x[np.where(gray_x == 76)] = 1 # weed = 76
    ## 76 and 150 pixel value can be check ed in GIMP software
    gray_x[np.where(gray_x == 150)] = 2 # crop = 150
    rgb_gray_scale = np.zeros_like(x)
    rgb_gray_scale[:, :, 0] = gray_x
    rgb_gray_scale[:, :, 1] = gray_x
    rgb_gray_scale[:, :, 2] = gray_x
    i += 1
    # print(gray_x.shape)
    cv2.imwrite(os.path.join(path,img),rgb_gray_scale)
print('there are {} images converted'.format(i))

