import cv2
import numpy as np
from matplotlib import pyplot as plt


def repairImage(posx, posy):
    img = cv2.imread("./upload/image.jpg", 1)
    img2 = img.copy()

    # # สร้างcropปลอม
    # crop = img2[posx : posx + 150, posy : posy + 150]
    # cv2.imwrite("./upload/crop.jpg", crop)

    mask = cv2.imread("./upload/imageEdit.jpg", 1)
    print(mask.shape)

    img2[posx : (posx + mask.shape[0]), posy : (posy + mask.shape[1])] = mask
    cv2.imwrite("./upload/repaired.jpg", img2)
    
    # cv2.imshow("Image", crop)
    # cv2.waitKey(0)


# repairImage("50,50")
