import cv2
import numpy as np

img = cv2.imread(r'C:\Users\Prashant\Desktop\HW3\emma.jpg', 0)
right_clicks = list()

# this method is called at mouse is right-click
def mouse_callback(event, x, y, flags, params):
    # right-click event value is 2
    if event == 2:
        global right_clicks

        # store the coordinates of the right-click event
        right_clicks.append([x, y])
        print(right_clicks)


def pixel_coord(image):

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Image', mouse_callback)
    cv2.imshow('Image', image)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

def transform_matrix():

    orig_right_clicks = np.array([[143, 485, 259],
                                  [399, 571, 692],
                                  [1, 1, 1]])
    new_right_clicks = np.array([[801, 1601, 1399],
                                 [391, 471, 656],
                                 [1, 1, 1]])

    xinv = np.linalg.inv(np.matrix(orig_right_clicks))
    xz = new_right_clicks.dot(xinv)
    print(np.matrix.round(xz))

transform_matrix()

# pixel_coord(img)