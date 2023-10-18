import cv2
import numpy as np
from scipy import ndimage


def robert(file_path, save_path, threshold):
    roberts_cross_v = np.array([[1, 0], [0, -1]])

    roberts_cross_h = np.array([[0, 1], [-1, 0]])

    # Read the original image
    img = cv2.imread(file_path)  
    # converting because opencv uses BGR as default
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.imread(file_path, 0).astype("float64")
    img /= 255.0
    vertical = ndimage.convolve(img, roberts_cross_v)
    horizontal = ndimage.convolve(img, roberts_cross_h)

    edged_img = np.sqrt(np.square(horizontal) + np.square(vertical))
    edged_img *= 255
    cv2.imwrite(save_path, edged_img)

    edge_detect(RGB_img, edged_img, threshold, save_path)

def edge_detect(color_img, edges, threshold, save_path):
    edge_pixels = {}
    for y in range(1, edges.shape[0] - 1):
        found_left = False
        right = 0

        for x in range(1, edges.shape[1] - 1):
            if edges[y][x] > threshold:
                right = x
                if not found_left:
                    edge_pixels[y] = [x]
                    found_left = True

        if y in edge_pixels:
            edge_pixels[y].append(right)

    # print(edge_pixels)

    print(color_img)
    for y in range(0, edges.shape[0]):
        for x in range(0, edges.shape[1]):
            if y in edge_pixels and x >= edge_pixels[y][0] and x <= edge_pixels[y][1]:
                color_img[y][x] = [255, 255, 255]
            else:
                color_img[y][x] = [0, 0, 0]

    cv2.imwrite(save_path, color_img)

robert("./sample/apple_85.jpg", "test_apple_85.jpg", 20)
