import cv2
import glob

def resizer(folder):
    img_file = glob.glob(folder)
    for file in img_file:
        img = cv2.imread(file)
        img = cv2.resize(img,(345,230))
        cv2.imwrite(file,img)


if __name__ == "__main__":

    folder = 'E:\\PhD_annotated\\gif\\add_weight1\\*'
    resizer(folder)

