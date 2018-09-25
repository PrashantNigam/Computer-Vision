import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

img = cv2.imread(r'C:\Users\Prashant\Desktop\HW3\emma.jpg', 0)
NEAREST_NEIGHBOUR = 1
BILINEAR = 2
# Translation parameters
tx = 20
ty = 30
# Rotation parameters
theta = math.pi/6
# Scaling parameters
sx = 1.2
sy = 1.3
# Shearing parameters
shx = 0.1
shy = 0.2

m_tran = np.array([[1, 0, tx],
                   [0, 1, ty],
                   [0, 0, 1]])

m_rot = np.array([[math.cos(theta), math.sin(theta), 0],
                  [-(math.sin(theta)), math.cos(theta), 0],
                  [0, 0, 1]])

m_scale = np.array([[sx, 0, 0],
                    [0, sy, 0],
                    [0, 0, 1]])

m_shear = np.array([[1, shx, 0],
                    [shy, 1, 0],
                    [0, 0, 1]])

m_scal_shear = np.array([[2, shx, 0],
                         [shy, 1, 0],
                         [0, 0, 1]])

aff = np.array([[sx * math.cos(theta), shx * math.sin(theta), tx],
                [-shy * (math.sin(theta)), sy * math.cos(theta), ty],
                [0, 0, 1]])


def nn_affine_transform(img, matrix, x, y):

    height = img.shape[0]
    width = img.shape[1]
    target_image = np.zeros((height + x, width + y))

    for y in range(height):
        for x in range(width):
            pix = img[y][x]
            img_mat = np.array([[x], [y], [1]])
            new = matrix.dot(img_mat)
            a, b = new[0], new[1]
            a, b = int(a[0]), int(b[0])
            target_image[b][a] = pix
    return target_image


def bilinear_affine_transform(img, matrix, x, y):

    height = img.shape[0]
    width = img.shape[1]
    new_img = np.zeros((height + x, width + y))

    for y in range(height):
        for x in range(width):
            pix = img[y][x]

            img_mat = np.array([[x],
                                [y],
                                [1]])

            new = matrix.dot(img_mat)

            a, b = new[0], new[1]
            bilinear_interpolation(a, b)
            # Removing parenthesis from values of new location
            a, b = int(a[0]), int(b[0])
            new_img[b][a] = pix

    return new_img


def bilinear_interpolation(x, y):

    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)

    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, img.shape[1] - 1);
    y0 = np.clip(y0, 0, img.shape[0] - 1);

    x1 = np.clip(x1, 0, img.shape[1] - 1);
    y1 = np.clip(y1, 0, img.shape[0] - 1);

    i1 = img[y0, x0]
    i2 = img[y1, x0]
    i3 = img[y0, x1]
    i4 = img[y1, x1]

    j1 = (x1 - x) * (y1 - y)
    j2 = (x1 - x) * (y - y0)
    j3 = (x - x0) * (y1 - y)
    j4 = (x - x0) * (y - y0)

    return (i1 * j1) + (i2 * j2) + (i3*j3) + (i4*j4)


def translation(int_type):

    if int_type == NEAREST_NEIGHBOUR:
        target_img = nn_affine_transform(img, m_tran, 100, 100)
        tag = "NEAREST_NEIGHBOUR"
    elif int_type == BILINEAR:
        target_img = bilinear_affine_transform(img, m_tran, 100, 100)
        tag = "BILINEAR"
    else:
        return

    original = plt.figure(figsize=(7, 5))
    original = plt.imshow(img, cmap='gray')
    original = plt.title("Original Image")
    plt.show(original)
    transformed = plt.figure(figsize=(8, 6))
    transformed = plt.imshow(target_img, cmap='gray')
    transformed = plt.title("Translated Image - " + tag)
    plt.show(transformed)


def rotation(int_type):

    if int_type == NEAREST_NEIGHBOUR:
        target_img = nn_affine_transform(img, m_rot, 1000, 1000)
        tag = "NEAREST_NEIGHBOUR"
    elif int_type == BILINEAR:
        target_img = bilinear_affine_transform(img, m_rot, 1000, 1000)
        tag = "BILINEAR"
    else:
        return

    original = plt.figure(figsize=(7, 5))
    original = plt.imshow(img, cmap='gray')
    original = plt.title("Original Image")
    plt.show(original)

    transformed = plt.figure(figsize=(8, 6))
    transformed = plt.imshow(target_img, cmap='gray')
    transformed = plt.title("Rotated Image - " + tag)
    plt.show(transformed)


def scaling(int_type):

    if int_type == NEAREST_NEIGHBOUR:
        target_img = nn_affine_transform(img, m_scale, 600, 600)
        tag = "NEAREST_NEIGHBOUR"
    elif int_type == BILINEAR:
        target_img = bilinear_affine_transform(img, m_scale, 600, 600)
        tag = "BILINEAR"
    else:
        return

    original = plt.figure(figsize=(7, 5))
    original = plt.imshow(img, cmap='gray')
    original = plt.title("Original Image")
    plt.show(original)

    transformed = plt.figure(figsize=(8, 6))
    transformed = plt.imshow(target_img, cmap='gray')
    transformed = plt.title("Scaled Image - " + tag)
    plt.show(transformed)


def shearing(int_type):

    if int_type == NEAREST_NEIGHBOUR:
        target_img = nn_affine_transform(img, m_shear, 1000, 1000)
        tag = "NEAREST_NEIGHBOUR"
    elif int_type == BILINEAR:
        target_img = bilinear_affine_transform(img, m_shear, 1000, 1000)
        tag = "BILINEAR"
    else:
        return
    original = plt.figure(figsize=(7, 5))
    original = plt.imshow(img, cmap='gray')
    original = plt.title("Original Image")
    plt.show(original)
    transformed = plt.figure(figsize=(8, 6))
    transformed = plt.imshow(target_img, cmap='gray')
    transformed = plt.title("Sheared Image - " + tag)
    plt.show(transformed)


def affine(int_type):

    if int_type == NEAREST_NEIGHBOUR:
        target_img = nn_affine_transform(img, aff, 1600, 1600)
        tag = "NEAREST_NEIGHBOUR"
    elif int_type == BILINEAR:
        target_img = bilinear_affine_transform(img, aff, 1600, 1600)
        tag = "BILINEAR"
    else:
        return
    original = plt.figure(figsize=(7, 5))
    original = plt.imshow(img, cmap='gray')
    original = plt.title("Original Image")
    plt.show(original)
    transformed = plt.figure(figsize=(8, 6))
    transformed = plt.imshow(target_img, cmap='gray')
    transformed = plt.title("Affine Transform - " + tag)
    plt.show(transformed)


def main():

    translation(NEAREST_NEIGHBOUR)
    rotation(NEAREST_NEIGHBOUR)
    scaling(NEAREST_NEIGHBOUR)
    shearing(NEAREST_NEIGHBOUR)
    affine(NEAREST_NEIGHBOUR)

    translation(BILINEAR)
    rotation(BILINEAR)
    scaling(BILINEAR)
    shearing(BILINEAR)
    affine(BILINEAR)
main()