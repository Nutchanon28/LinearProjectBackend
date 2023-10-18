import numpy as np
import cv2


def show_image():
    # Set up data to send to mouse handler
    data = {}
    img = cv2.imread("image.png", 1)
    print("read success!!")

    cv2.imshow("Image", img)
    cv2.waitKey(0)

    # Convert array to np.array in shape n,2,2
    points = np.uint16(data['lines'])

    return "Hello from the hell"

def show_monotone_image():
    # Set up data to send to mouse handler
    data = {}
    img = cv2.imread("image.png", 0)
    print("read success!!")

    cv2.imshow("Image", img)
    cv2.waitKey(0)

    # Convert array to np.array in shape n,2,2
    points = np.uint16(data['lines'])

    return "Hello from the hell"




# def mouse_handler(event, x, y, flags, data, btn_down):
    

#     if event == cv2.EVENT_LBUTTONUP and btn_down:
#         #if you release the button, finish the line
#         btn_down = False
#         data['lines'][0].append((x, y)) #append the second point
#         # cv2.circle(data['im'], (x, y), 3, (0, 0, 0),2)
#         cv2.rectangle(data['im'], data['lines'][0][0], (x,y), (0,0,0),1)
#         cv2.imshow("Image", data['im'])

#     elif event == cv2.EVENT_MOUSEMOVE and btn_down:
#         #thi is just for a ine visualization
#         image = data['im'].copy()
#         cv2.rectangle(image, data['lines'][0][0], (x,y), (0,0,0),1)
#         cv2.imshow("Image", image)

#     elif event == cv2.EVENT_LBUTTONDOWN:
#         btn_down = True
#         data['lines'][0] = [(x, y)] #prepend the point
#         # cv2.circle(data['im'], (x, y), 3, (0, 0, 0),2)
#         cv2.imshow("Image", data['im'])

  