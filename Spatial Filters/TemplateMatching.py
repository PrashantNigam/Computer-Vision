import numpy as np
import cv2
import math


def template_matching():

    img = cv2.imread(r'C:\Users\Prashant\Desktop\CV imgs\shapes-bw.jpg', 0)
    cv2.imshow('Original Image', img)
    cv2.waitKey(10000)

    template = cv2.imread(r'C:\Users\Prashant\Desktop\CV imgs\1ast.jpg', 0)
    cv2.imshow('Template', template)
    cv2.waitKey(10000)

    template = reduce_intensity_by_mean(template)
    img = correlation(template, img)
    cv2.imshow('Correlation Image', img)
    cv2.waitKey(10000)

    img = do_thresholding(img)
    cv2.imshow('Thresholded Image', img)
    cv2.waitKey(10000)

    img = peak_detection(img)
    cv2.imshow('Peak-Detected Correlation Image', img)
    cv2.waitKey(10000)


def reduce_intensity_by_mean(image):
    return image - np.mean(image)


def correlation(mask, arr):

    temp = np.zeros((len(arr), len(arr[0])))
    arrRows = len(arr)
    arrCols = len(arr[0])
    maskRows = len(mask)
    maskCols = len(mask[0])
    padH = math.floor(maskRows/2)
    padV = math.floor(maskCols/2)
    for i in range(padH, (arrRows - padH)):
        for j in range(padV, (arrCols - padV)):
            sub_matrix = arr[i - padH:i + padH, j - padV:j + padV]
            sub_matrix = reduce_intensity_by_mean(sub_matrix)
            temp[i][j] = np.mean(sub_matrix * mask)
    temp = smoothing(temp)
    return temp


def do_thresholding(img):

    threshold = np.mean(img)
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] < threshold:
                img[i][j] = 0
    return img


def main():
    template_matching()


def smoothing(img):

    size = 10
    kernel = np.ones((size,size),np.float32)/(size*size)
    array = []
    for j in range(size):

        temp = np.copy(img)
        temp = np.roll(temp, j - 1, axis=0)
        for i in range(size):

            temp_X = np.copy(temp)
            temp_X = np.roll(temp_X, i - 1, axis=1) * kernel[j, i]
            array.append(temp_X)

    array = np.array(array)
    array_sum = np.sum(array, axis=0)
    for i in range(len(array_sum)):
        for j in range(len(array_sum[0])):
            img.itemset((i,j), array_sum[i][j])
    return img


def peak_detection(image):

    kernel = np.array([[0,-1,0], [-1,4,-1], [0,-1,0]], np.int32)
    m, n = kernel.shape
    y, x = image.shape
    y = y - m + 1
    x = x - m + 1
    convoluted = np.zeros((y,x))
    for i in range(y):
        for j in range(x):
            convoluted[i][j] = np.sum(image[i:i+m, j:j+m]*kernel)
    return convoluted

main()