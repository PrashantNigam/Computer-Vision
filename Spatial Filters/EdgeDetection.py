import numpy as np
import cv2

'''
The Process of Canny edge detection algorithm can be broken down to 5 different steps:

1. Apply Gaussian filter to smooth the image in order to remove the noise
2. Find the intensity gradients of the image
(cite: from Wikipedia for reference)
'''


def edge_detection():

    # Step 1
    img = smoothing()
    # Step 2
    get_intensity_gradients(img)


'''
This methods calculates partial derivatives vectors and returns its magnitude and direction.
'''
def get_intensity_gradients(img):

    # Pass array and dataType in this method
    # Kernel for Gradient in x-direction
    Mx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32)
    # Kernel for Gradient in y-direction
    My = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32)
    # Apply kernels to the image
    Ix = convolve(img, Mx)
    cv2.imshow('Image Derivative - X (dx)', Ix)
    cv2.waitKey(10000)

    Iy = convolve(img, My)
    cv2.imshow('Image Derivative - Y (dy)', Iy)
    cv2.waitKey(10000)

    magnitude = np.hypot(Ix, Iy)
    cv2.imshow('Edge Map', magnitude)
    cv2.waitKey(10000)

    direction = np.arctan2(Iy, Ix)
    cv2.imshow('Orientation Map', direction)
    cv2.waitKey(10000)
    return magnitude, direction


def convolve(image, kernel):

    m, n = kernel.shape
    y, x = image.shape
    y = y - m + 1
    x = x - m + 1
    convoluted = np.zeros((y,x))
    for i in range(y):
        for j in range(x):
            convoluted[i][j] = np.sum(image[i:i+m, j:j+m]*kernel)
    return convoluted


def main():
    edge_detection()

def smoothing(n=3):

    kernel = np.ones((n, n))
    kernel = kernel / np.sum(kernel)
    array = []
    img = cv2.imread(r'C:\Users\Prashant\Desktop\CV imgs\capitol.jpg', 0)
    cv2.imshow('Original Image', img)
    cv2.waitKey(10000)
    for j in range(n):

        temp = np.copy(img)
        temp = np.roll(temp, j - 1, axis=0)
        for i in range(n):

            temp_X = np.copy(temp)
            temp_X = np.roll(temp_X, i - 1, axis=1) * kernel[j, i]
            array.append(temp_X)

    array = np.array(array)
    array_sum = np.sum(array, axis=0)
    for i in range(len(array_sum)):
        for j in range(len(array_sum[0])):
            img.itemset((i,j), array_sum[i][j])

    cv2.imshow('After Smoothing with Size ' + str(n) + ' kernel', img)
    cv2.waitKey(10000)
    return img

main()