import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

color_img = cv.imread("./sample/apple_85.jpg")
img = cv.imread("./sample/apple_85.jpg", cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
edges = cv.Canny(img, 100, 200)

cv.imwrite("canny_edges.jpg", edges)
print(edges.shape)
print(edges)

# edges.shape (row, col)
edge_pixels = {}
for y in range(1, edges.shape[0] - 1):
    found_left = False
    right = 0

    for x in range(1, edges.shape[1] - 1):
        if edges[y][x] == 255:
            right = x
            if not found_left:
                edge_pixels[y] = [x]
                found_left = True
        # print(y, x)

    if y in edge_pixels:
        edge_pixels[y].append(right)

print(edge_pixels)

for y in range(1, edges.shape[0] - 1):
    if y not in edge_pixels:
        continue

    for x in range(1, edges.shape[1] - 1):
        if x == edge_pixels[y][0] or x == edge_pixels[y][1]:
            color_img[y][x] = [102, 0, 0]
        elif x >= edge_pixels[y][0] and x <= edge_pixels[y][1]:
            color_img[y][x] = [0, 102, 0]

cv.imwrite("canny_object.jpg", color_img)

# plt.subplot(121), plt.imshow(img, cmap="gray")
# plt.title("Original Image"), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(edges, cmap="gray")
# plt.title("Edge Image"), plt.xticks([]), plt.yticks([])
# plt.show()


# wrong_extension = ["xml", "DS_Store"]
# directory = "../test"
# saved_directory = "../canny1"

# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
#     if os.path.isfile(f) and filename.split(".")[1] not in wrong_extension:
#         print(f)
#         img = cv.imread(f"{directory}/{filename}", cv.IMREAD_GRAYSCALE)
#         edges = cv.Canny(img, 100, 200)

#         cv.imwrite(f"{saved_directory}/{filename}", edges)