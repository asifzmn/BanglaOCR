import string
import collections
import cv2
import plotly.plotly as py
import plotly.tools as tls

import matplotlib.pyplot as plt

from BasicImplementation import Preprocessing, DisplayImage, PreprocessingAdaptive
from PaperImplementation import LineSeparation, HorizontalProjectionProfile


def PrepareWordsLebel():

    file = open("Newspaper Sample/1.txt", "r")

    paperArticle = file.read()
    lines = paperArticle.split('\n')
    wordsArray = []

    for line in lines:
        wordsInLine = line.split(' ')
        wordsArray.append(wordsInLine)

    return wordsArray

def PrepareWordsImageData():
    image = cv2.imread("Newspaper Sample/1.png", 0)
    processedImage = PreprocessingAdaptive(image,5)[:500]
    lines = LineSeparation(processedImage, 0)
    headerPositions, headerRange = HorizontalProjectionProfile(processedImage, lines)
    print(headerPositions)
    DisplayImage(processedImage)

    return



if __name__ == '__main__':

    wordsArray = PrepareWordsLebel()
    WordImageArray = PrepareWordsImageData()

    # book = book.translate(str.maketrans('', '', string.punctuation))
    # book = book.replace('“','').replace('।','').replace('”','').replace('\n',' ').split(' ')
    # print(paperArticle)

    # counter = collections.Counter(book)
    # print(counter)