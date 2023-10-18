import cv2

def sorbel(file_path, save_path, threshold):
    # Read the original image
    img = cv2.imread(file_path)  
    # converting because opencv uses BGR as default
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # converting to gray scale
    gray = cv2.cvtColor(RGB_img, cv2.COLOR_BGR2GRAY)

    # remove noise
    img = cv2.GaussianBlur(gray,(3,3),0)

    # convolute with sobel kernels
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y
    sobel = sobelx + sobely

    edge_detect(RGB_img, sobel, threshold, save_path)

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

sorbel("./sample/apple_85.jpg", "test_apple_85.jpg", 30)
