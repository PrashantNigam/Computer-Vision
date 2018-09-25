import cv2
import numpy as np
import math
from matplotlib import pyplot as plot


def main():

    # img = cv2.imread(r'C:\Users\Prashant\Desktop\CV ans\B3\Matched\matched.jpg', 0)
    # calcHisto(img, 256)
    '''
        Part B1 = load an image from the local directory and pass it calcHisto(img, n)
        Part B2 = load an image from the local directory and pass it equalise(img)
        Part B3 = load 2 images from the local directory and pass them to histogramMatching(source, reference)
        source is the poor image and reference is the rich image feature and contrast wise
    '''


def calcHisto(image, n):

    width = n
    row, col = image.shape
    y = np.zeros(width, np.uint64)
    for i in range(0,row):
        for j in range(0,col):
            y[round(image[i, j])] += 1
    x = np.arange(0, width)
    plot.bar(x, y, color="gray", align="center")
    plot.show()
    plot.close()
    normalise(x, y)
    return x, y


def normalise(x, y):

    weights = np.ones_like(y) / float(len(y))
    plot.bar(x, weights, color="gray", align="center")
    plot.plot(weights)
    plot.show()
    plot.close()
    createPDF(x, y)


def createPDF(x, y):

    sumFreq = np.sum(y)
    pdf = np.divide(y, sumFreq)
    plot.plot(pdf)
    plot.show()
    plot.close()
    createCDF(x, y)
    return pdf


def createCDF(x, y):

    cdf = np.cumsum(y)
    plot.plot(cdf)
    plot.show()
    plot.close()
    return cdf


def equalise(img):

    height = img.shape[0]
    width = img.shape[1]
    pixels = width * height
    histogram = getHistogram(img)
    cumulativeHistogram = getCumulativeHistogram(histogram)

    for i in np.arange(height):
        for j in np.arange(width):
            a = img.item(i, j)
            b = math.floor(cumulativeHistogram[a] * 255.0 / pixels)
            img.itemset((i, j), b)

    cv2.imshow('equalised', img)
    cv2.waitKey(10000)
    calcHisto(img, 256)


def histogramMatching(poor, rich):

    height = poor.shape[0]
    width = poor.shape[1]
    pixels = width * height
    height_ref = rich.shape[0]
    width_ref = rich.shape[1]
    pixels_ref = width_ref * height_ref
    hist = getHistogram(poor)
    histRef = getHistogram(poor)
    cumHist = getCumulativeHistogram(hist)
    cumHistRef = getCumulativeHistogram(histRef)
    prob_cum_hist = cumHist / pixels
    prob_cum_hist_ref = cumHistRef / pixels_ref
    bins = 256
    new_values = np.zeros((bins))
    for a in np.arange(bins):
        j = bins - 1
        while True:
            new_values[a] = j
            j = j - 1
            if j < 0 or prob_cum_hist[a] > prob_cum_hist_ref[j]:
                break

    for i in np.arange(height):
        for j in np.arange(width):
            a = poor.item(i, j)
            b = new_values[a]
            poor.itemset((i, j), b)
    cv2.imshow('matched', poor)
    cv2.waitKey(10000)
    calcHisto(poor, 256)

def getHistogram(img):

    height = img.shape[0]
    width = img.shape[1]

    hist = np.zeros((256))

    for i in np.arange(height):
        for j in np.arange(width):
            a = img.item(i, j)
            hist[a] += 1

    return hist


def getCumulativeHistogram(histogram):

    cum_hist = histogram.copy()
    for i in np.arange(1, 256):
        cum_hist[i] = cum_hist[i - 1] + cum_hist[i]

    return cum_hist

main()
