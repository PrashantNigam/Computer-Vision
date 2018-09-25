import numpy as np
import cv2


def smoothing(n=3):

    kernel = np.ones((n, n))
    kernel = kernel / np.sum(kernel)
    array = []
    img = cv2.imread(r'C:\Users\Prashant\Desktop\CV imgs\capitol.jpg', 0)
    cv2.imshow('Before Smoothing', img)
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


def main():

    smoothing(3)
    smoothing(5)


main()
