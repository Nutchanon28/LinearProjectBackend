import cv2
import numpy as np

def prewitt(file_path, save_path, threshold):
    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray,(3,3),0)

    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
    img_prewitt = img_prewittx + img_prewitty
    # cv2.imshow("Prewitt", img_prewittx + img_prewitty)
    cv2.imwrite("./upload/mask_edge.jpg", img_prewitt)
    edge_detect(img, img_prewitt, threshold, save_path)

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

prewitt("./upload/crop.jpg", "./upload/mask.jpg", 70)

