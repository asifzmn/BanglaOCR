import collections
import string
import cv2
import  numpy as np
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize, thin
from skimage.util import invert
from PaperImplementation import Preprocessing, LineSeparation, HorizontalProjectionProfile, BaseLineDetection, \
    CandidateColumns, DisplayImage

if __name__ == '__main__':
    file = open("Krishnokanter Will/1", "r")

    book = file.read()
    book = book.translate(str.maketrans('', '', string.punctuation))
    book = book.replace('“','').replace('।','').replace('”','').replace('\n',' ').split(' ')
    print(book)

    counter = collections.Counter(book)
    print(counter)


    # image = invert(cv2.imread('Krishnokanter Will/scr1.png', 0))
    # processedImage = Preprocessing(image)
    # # processedImage = Preprocessing(image)
    # lines = LineSeparation(processedImage, 1)
    # headerPositions = HorizontalProjectionProfile(processedImage, lines)
    # baseLines = BaseLineDetection(processedImage, lines, headerPositions)
    # rowColumnList = CandidateColumns(processedImage, headerPositions, lines)
    #
    # DisplayImage(processedImage)

    exit()
