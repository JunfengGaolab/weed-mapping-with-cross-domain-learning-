import numpy as np
import cv2

def drawRec(img, ann):
    """
    :param img: original image for prediction
    :param ann: annotation or labelled or predicted mask image from the trained model
    :return: rectangle box for segments in the annotation or mask image
    """
    img = np.copy(img)
    original_ann = np.copy(ann)
    ## todo be mindful with the reading channel order of opencv model
    weed_mask = np.zeros(ann.shape[:2])
    crop_mask = np.zeros(ann.shape[:2])
    weed_mask = ann[:, :, 2]
    ret, thresh_weed = cv2.threshold(weed_mask, 127, 255, cv2.THRESH_BINARY)
    _, cnts_weed, _ = cv2.findContours(thresh_weed, 1, 2)
    nb_weed = 0
    weed_color = (0, 0, 255)
    for contour in cnts_weed:
        if cv2.contourArea(contour) > 50:  ## setting area threshold for showing bounding box
            nb_weed += 1
            x_weed, y_weed, w_weed, h_weed = cv2.boundingRect(contour)
            start_point = (x_weed, y_weed)
            end_point = (x_weed + w_weed, y_weed + h_weed)
            cv2.rectangle(img, start_point, end_point, weed_color, 3)
    print('There are {} weeds detected'.format(nb_weed))
### crop bounding boxing drawing
    # crop_mask = ann[:, :, 1]
    # ret, thresh_crop = cv2.threshold(crop_mask, 127, 255, cv2.THRESH_BINARY)
    # _, cnts_crop, _ = cv2.findContours(thresh_crop, 1, 2)
    # nb_crop = 0
    # crop_color = (0, 255, 0)
    # for contour in cnts_crop:
    #     if cv2.contourArea(contour) > 100:  ## setting area threshold for showing bounding box
    #         nb_crop += 1
    #         x_crop, y_crop, w_crop, h_crop = cv2.boundingRect(contour)
    #         start_point = (x_crop, y_crop)
    #         end_point = (x_crop + w_crop, y_crop + h_crop)
    #         cv2.rectangle(img, start_point, end_point, crop_color, 2)
    # print('There are {} crop plants detected'.format(nb_crop))
    return img


if __name__ == '__main__':
    img = cv2.imread('E:\\PhD_annotated\\random_test\\IMG_0028.JPG')
    revert_ann = cv2.imread('E:\\PhD_annotated\\random_test_result\\IMG_0028_predict.jpg')
    cv2.imshow('res', drawRec(img, revert_ann))
    cv2.waitKey(0)
