import numpy as np
import cv2
import matplotlib.pyplot as plt


def hough_line(img):

    theta_axis_divisions = np.deg2rad(np.arange(-180.0, 180.0))
    width, height = img.shape
    len_diagonal = np.ceil(np.sqrt(width * width + height * height))
    rho_axis_divisions = np.linspace(-len_diagonal,
                                     len_diagonal,
                                     len_diagonal * 2.0)

    cos_theta = np.cos(theta_axis_divisions)
    sin_theta = np.sin(theta_axis_divisions)
    thetas_range = len(theta_axis_divisions)

    accumulator = np.zeros((2 * int(len_diagonal), thetas_range))
    vertical_indices, horizontal_indices = np.nonzero(img)

    # Vote in the hough accumulator
    for i in range(len(horizontal_indices)):
        x = horizontal_indices[i]
        y = vertical_indices[i]

        for idx_theta in range(thetas_range):

            rho = round(x * cos_theta[idx_theta] + y * sin_theta[idx_theta]) + len_diagonal
            accumulator[int(rho), idx_theta] += 1

    return accumulator, theta_axis_divisions, rho_axis_divisions


def trigger():

    processed_image = cv2.imread(r'C:\Users\Prashant\Desktop\CV_NEW\2\processed_1.jpg', 0) # edge image
    accumulator, thetas, rhos = hough_line(processed_image)
    plt.figure(figsize=(15, 10))
    plt.title("Accumulator Grid")
    plt.imshow(accumulator, cmap='gray')

    filtered = gaussian_filter(accumulator)
    plt.figure(figsize=(15, 10))
    plt.title("Accumulator Grid After Gaussian Smoothing")
    plt.imshow(filtered, cmap='gray')

    max = filtered.max()
    metadata = []
    height, width = filtered.shape
    for rho in range(height):
        for theta in range(width):
            if filtered[rho, theta] == max:
                metadata.append([rho, theta, filtered[rho, theta]])
    print(metadata)

    theta = [45.041, 45.041, 0.01723, 0.01676]
    rho = [157, 189, 205, 215]
    edge = np.copy(processed_image)

    for line in g:
        for i in range(len(theta)):
            a = np.cos(theta[i])
            b = np.sin(theta[i])
            x0 = a * rho[i]
            y0 = b * rho[i]
            x1 = int(x0 - b*1000)
            y1 = int(y0 + a*1000)
            x2 = int(x0 + b*1000)
            y2 = int(y0 - a*1000)
            cv2.line(edge, (x1, y1), (x2, y2), (60, 60, 60), 1)

    plt.figure(20, 15)
    plt.title("Final Detected lines")
    plt.imshow(edge)


def convolution(img, filter):

    img_height = img.shape[0]
    img_width = img.shape[1]

    filter_height = filter.shape[0]
    filter_width = filter.shape[1]

    img_height_dash = (filter_height-1)/2
    img_width_dash = (filter_width-1)/2

    result = np.zeros((img_height, img_width))
    for i in np.arange(img_height_dash, (img_height - img_height_dash)):
        for j in np.arange(img_width_dash, (img_width - img_width_dash)):
            sum_values = 0
            for k in np.arange(-img_height_dash, (img_height_dash + 1)):
                for l in np.arange(-img_width_dash, (img_width_dash + 1)):
                    img_pixel = img[i+k, j+l]
                    filter_pxl = filter[img_height_dash + k, img_width_dash + l]
                    sum_values += (filter_pxl*img_pixel)
            result[i, j] = sum_values
    return result


def gaussian_filter(img):

    height = img.shape[0]
    width = img.shape[1]
    # Sobel Operator source: https://en.wikipedia.org/wiki/Sobel_operator
    G = np.array([[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]])
    G = G/np.sum(G)
    img_g = convolution(img, G)
    return img_g

trigger()
