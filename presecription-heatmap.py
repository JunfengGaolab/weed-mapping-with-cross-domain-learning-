import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from pandas import DataFrame
def draw_grid(img, line_color=(0, 255, 0), thickness=1, type_=cv2.LINE_AA, pxstep=50):
    '''(ndarray, 3-tuple, int, int) -> void
    draw gridlines on img
    line_color:
        BGR representation of colour
    thickness:
        line thickness
    type:
        8, 4 or cv2.LINE_AA
    pxstep:
        grid line frequency in pixels
    '''
    x = pxstep
    y = pxstep
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pxstep

    return img
def count_weed_grid(image, grid_size = 100):
    count_weed = 0
    count_grid = 0
    for x in range(0, image.shape[1], grid_size):
        if x+grid_size < image.shape[1]:
            for y in range(0, image.shape[0], grid_size):
                if y+grid_size<image.shape[0]:
                    count_grid += 1
                    if np.sum(image[x:x+grid_size, y:y+grid_size]) >100:
                        count_weed += 1
    weed_ratio = count_weed/count_grid
    free_weed_ratio = 1 - weed_ratio
    count_free_grid = count_grid - count_weed
    return count_free_grid,count_weed, weed_ratio, free_weed_ratio*100




raw_image = cv2.imread('original-roi-orthomosaic.png')
raw_image_GT = cv2.imread('orthomosaic-GT.png') ### Ground truth
# raw_image_prediction = cv2.imread('orthomosaic-weighted-segunet-prediction.jpg')
raw_image_prediction = cv2.imread('orthomosaic-GT.png')
print(raw_image.shape[:2])
print(raw_image_prediction.shape)





height, width, _ = raw_image_prediction.shape
bin = np.zeros((height, width, 1), np.uint8)


gray_scale= cv2.cvtColor(raw_image_prediction, cv2.COLOR_BGR2GRAY)

# th = cv2.threshold(gray_scale, 100, 255, cv2.THRESH_BINARY_INV)[1]
# intersection = gray_scale*th
th = cv2.inRange(gray_scale, 50, 90) # inrange segmentation
print (th.shape)



# remove the small blobs
nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(th)
sizes = stats[:, -1]
sizes = sizes[1:]
nb_blobs -= 1
min_size = 150
th_result = np.zeros((th.shape), np.uint8)

for blob in range(nb_blobs):
    if sizes[blob] >= min_size:
        # see description of im_with_separated_blobs above
        th_result[im_with_separated_blobs == blob + 1] = 255

a,b,c,d = count_weed_grid(th_result,50)
print ("GT's total number of free weed grids {}, weed grids {}, weed ratio {} and free_weed_ratio are {}".format(a,b,c,d))

# TODO: draw a line for spraing rate with regard to spraying resolution (grid)
def draw_line(image):
    list_x, list_y = [], []
    for i in range (1, 251, 5):
        list_x.append(i)
        a, b, c, d = count_weed_grid(image, i)
        list_y.append(d)
    array_x = np.array(list_x)
    array_y = np.array(list_y)
    # a_line, b_line = np.polyfit(array_x, array_y, 1)

    a_line, b_line, r_value, p_value, std_err = scipy.stats.linregress(array_x, array_y)
    print(r_value)
    print(r_value**2)
    print('r value is---{}'.format(r_value**2))
    plt.scatter(list_x, list_y, marker='o', sizes=[i for i in list_y],
            cmap='viridis')

    plt.plot(array_x, a_line * array_x + b_line, color = 'r', linestyle='--', linewidth=2)
    plt.text(147, 82, 'y = ' + '{:.3f}'.format(b_line) + '{:.3f}'.format(a_line) + 'x' + '\n' + 'R^2 = {:.3f}'.format(r_value**2), size=14)
    # plt.text(107,140,'R^2 = '+str('{:.3f}'.format(r_value**2)), size=14) it does not work
    # plt.legend()
    plt.xlabel('Grid size')
    plt.ylabel('Spraying saving percentage (%)')
    # plt.show()
    plt.savefig('spraying-spatial-r-gt.png', dpi=600)

draw_line(th_result)


blur = cv2.GaussianBlur(th_result, (15, 15), 11)
heatmap = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
raw_image1 = cv2.resize(raw_image, (2112,2416))
# superimposing the heatmap over the original image
print(raw_image1.shape)
print(heatmap.shape)

super_imposed_img = cv2.addWeighted(heatmap, 0.3, raw_image1, 1, 0)

# draw grid on the image
prescription_map = draw_grid(heatmap, line_color=(255, 255, 255), thickness=2, type_=cv2.LINE_AA, pxstep=100)

save_name = 'save_prescription_map.jpg'
save_name1 = 'save_heatmap.jpg'
cv2.imwrite(save_name1, super_imposed_img)
cv2.imwrite(save_name, prescription_map)

cv2.namedWindow('heatmap', cv2.WINDOW_NORMAL)
cv2.imshow('heatmap',prescription_map)
cv2.waitKey(0)