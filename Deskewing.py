import sys
import cv2
import matplotlib.pyplot as plt
import numpy
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation as inter

from PaperImplementation import Preprocessing, LineSeparation, HorizontalProjectionProfile, BaseLineDetection,CandidateColumns, DisplayImage


def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score




if __name__ == '__main__':

    img = im.open('Images/camera.jpg')
    # img = im.open('scr1.png')
    # img = im.open('images.png')

    # convert to binary
    wd, ht = img.size
    pix = np.array(img.convert('1').getdata(), np.uint8)
    bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)


    # plt.imshow(bin_img, cmap='gray')
    # plt.savefig('binary.png')

    delta = 1
    limit = 5
    angles = np.arange(-limit, limit + delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        scores.append(score)

    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    # print('Best angle: {}',(best_angle))

    # correct skew
    data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
    # img = im.fromarray((255 * data).astype("uint8")).convert("RGB")
    img = im.fromarray((255 * data).astype("uint8"))
    # img.save('skew_corrected.png')

    print(np.shape(img))

    img = numpy.array(img)

    # for i in img:
        # for j in i:
        #     print(j)
        # print('-----------')

    # image = remove_noise_and_smooth(img)

    # imgh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    # kernel = np.ones((5, 5), np.uint8)
    # # gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    # DisplayImage(img)


    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    # processedImage = Preprocessing(img)
    # lines = LineSeparation(processedImage, 1)
    # headerPositions = HorizontalProjectionProfile(processedImage,lines)
    # baseLines = BaseLineDetection(processedImage,lines,headerPositions)
    # rowColumnList = CandidateColumns(processedImage,headerPositions,lines)

    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

    DisplayImage(img)
    # DisplayImage(processedImage)

