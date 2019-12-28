import collections
import os
import shutil
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from skimage.filters import threshold_local

from ImagePreparation import PrepareImage, DisplayImage, Preprocessing, reduceResolution, Saving, DisplayImages
from Optics import Clustering

# ['Bardidi' 'DarwanMaleeKathapokathan' 'DhyanOSakti' 'GopiChandrerGan'
#  'Gospelofmatthewi' 'NewTestament' 'OnlyWayTobeSaved' 'ParamarthaPrasanga'
#  'PurbobangoOHinduSamajh' 'SalomanerHitopodesh' 'SankhaDarsanam']

# ['Bardidi' 'BiswasGhatakMahon' 'DarwanMaleeKathapokathan' 'DhyanOSakti'
#  'GopiChandrerGan' 'Gospelofmatthewi' 'ManahShiksha' 'MatirPath'
#  'MohonerAgatobas' 'NewTestament' 'OnlyWayTobeSaved' 'ParamarthaPrasanga'
#  'SankhaDarsanam']

# [ 'BiswasGhatakMahon' 'ManahShiksha' 'MatirPath'
#  'MohonerAgatobas'  'SankhaDarsanam']



def TextLinesEnumarated(fullText):
    textLinesEnumarated = ''
    for i, l in enumerate(fullText): textLinesEnumarated += str(i) + ' ' + l + '\n'
    return textLinesEnumarated + "\n"


def OriginalLines(labelLocation):
    fullText = open(labelLocation).read().split("\n")[:-1:]
    originalLineCount = len(fullText)
    return fullText, originalLineCount


def WordLabeling(DataLoc):
    if not save: return
    labelFile = open("Groundtruth.txt", "a+")
    subfolders = [f.path for f in os.scandir(DataLoc) if f.is_dir()]
    for subFolder in subfolders:
        for c, wordFiles in enumerate([f for f in listdir(subFolder) if isfile(join(subFolder, f))]):
            finalName = str(c) + '.png'
            src = (subFolder + '/' + wordFiles)
            dst = (DataLoc + 'Ready/' + finalName)
            newPath = shutil.copy2(src, dst)
            labelFile.write(finalName + '@' + wordFiles[:-4] + '\n')


def LineLabeling(allFiles,locationLabel):
    if not save: return
    labelFile = open("Groundtruth.txt", "a+")
    for file in allFiles:
        splitted = file.split('_')
        textfile = '_'.join(splitted[:-1]) + '.txt'
        loc = int(splitted[2][1:].split('.')[0])
        x = OriginalLines(join(locationLabel, textfile))[0][loc - 1]
        labelFile.write(file + '@' + x + '\n')


def CompareClustWLine(file, opticsLine, originalLines, counts, lineTexts):
    dif.append(originalLines - opticsLine)
    # print(opticsLine-originalLines)
    if opticsLine > originalLines:
        counts[0] += 1
        # images = [[str(i), image] for i,image in enumerate([cv2.imread(LinePath + '/' + file + '/' + str(x) + '.png') for x in range(opticsLine)])]

    if opticsLine < originalLines:
        counts[1] += 1

    if opticsLine == originalLines:
        counts[2] += 1
        # for i in range(OpticsLine):
        # print(lineTexts[i])
        # DisplayImage(cv2.imread(LinePath+'/'+file+'/'+str(i)+'.png'))


def LineMatching(file, counts, mainLabelPath, LinePath):
    labelLocation = mainLabelPath + '/' + file
    segmentedLineDir = LinePath + '/' + file
    coloredImagePath = LinePath + '-Original/' + file + '.png'

    if not isfile(labelLocation) or not os.path.exists(segmentedLineDir) or not isfile(coloredImagePath): return

    allFiles = [f for f in listdir(segmentedLineDir)]
    opticsLine = len(allFiles)
    lineTexts, originalLineCount = OriginalLines(labelLocation)
    image = cv2.imread(coloredImagePath)
    # CompareClustWLine(file,opticsLine,originalLineCount,counts,lineTexts)
    print(file)
    print(opticsLine - originalLineCount)
    print(TextLinesEnumarated(lineTexts))
    print('image: ', opticsLine, 'text: ', originalLineCount)
    # print("Check Image?(1 to confirm)")
    # if(int(input())==1):
    if (True):
        # DisplayImage(reduceResolution(image,ratio=3)[0])
        if (opticsLine == originalLineCount):
            # print("Confirm Label With Equal/Missing Lines?(1 to confirm)")
            # if(int(input())==1):
            LineConfirmation(coloredImagePath, segmentedLineDir, lineTexts, ImageLineMap(opticsLine, originalLineCount))
            # Confirmation(coloredImagePath,segmentedLineDir,lineTexts,np.arange(originalLineCount))
        if (opticsLine < originalLineCount):
            print("Confirm Label With Unequal Lines?(1 to confirm)")
            if (int(input()) == 1):
                DisplayImage(reduceResolution(image, ratio=3)[0])
                LineConfirmation(coloredImagePath, segmentedLineDir, lineTexts,
                                 ImageLineMap(opticsLine, originalLineCount))


def LineConfirmation(coloredImagePath, segmentedLineDir, lineTexts, lineMap):
    os.rename(coloredImagePath, coloredImagePath[:-4:] + 'check' + coloredImagePath[-4::])

    for i, l in enumerate(lineMap):
        os.rename(segmentedLineDir + '/' + str(i) + '.png', segmentedLineDir + '/' + lineTexts[l] + '.png')

    return


def ImageLineMap(a, b):
    A = list(range(a))
    B = list(range(b))
    delList = []
    for i in range(b - a): delList.append(int(input()))
    for d in delList: B.remove(d)
    # return (np.vstack((A, B)))
    return np.array(B)




def WordSegmentation(fileName):
    image = np.transpose(PrepareImage(fileName))
    lineImages = Clustering(image, minSamples=15, xI=.03, minClusterSize=.025, scaling=1, word=0, p=1.5)
    fullLineImages = [(a, np.transpose(b, axes=[1, 0, 2])) for a, b in lineImages]
    return fullLineImages[:-1:], fullLineImages[-1::]


def WordConfirmation(coloredImagePath, segmentedLineDir, lineTexts, lineMap):
    os.rename(coloredImagePath, coloredImagePath[:-4:] + 'check' + coloredImagePath[-4::])

    for i, l in enumerate(lineMap):
        os.rename(segmentedLineDir + '/' + str(i) + '.png', segmentedLineDir + '/' + lineTexts[l] + '.png')

    return


def WordMatching(file, LinePath):
    chekckedFile = LinePath + '-Original/' + file + 'check.png'
    chekckedWordFile = LinePath + '/' + file
    if isfile(chekckedFile) and os.path.exists(chekckedWordFile):
        counts[3] += 1
        print(file)
        LineImageFiles = [f for f in listdir(chekckedWordFile) if isfile(join(chekckedWordFile, f))]

        for i, lineImagefile in enumerate(LineImageFiles):
            wordSaveLocation = LinePath + '-words/' + lineImagefile
            if os.path.exists(wordSaveLocation): continue
            words = lineImagefile.split(' ')[:-1:]
            individuals, full = WordSegmentation(chekckedWordFile + '/' + lineImagefile)
            DisplayImages(full)

            if len(words) == len(individuals):
                print(words)
                # DisplayImages(full)
                allow = True
                if allow:
                # if (int(input())==1):
                    individuals = [[words[w], l[1]] for w, l in enumerate(individuals)]
                    Saving(wordSaveLocation, individuals, save)
    return


def LineSegmentation(file, LinePath, mainImagePath, save):
    if os.path.exists(LinePath + '/' + file): return
    print(file)

    imageFilePath = mainImagePath + '/' + file[:-3] + 'jpeg'
    image = (PrepareImage(imageFilePath))
    if not save:return
    LineImages = Clustering(image, minSamples=30, xI=.15, minClusterSize=.005, scaling=15)

    Saving(LinePath + '/' + file, LineImages[:-1:], save)
    Saving(LinePath + '-Original', [(file, LineImages[-1][1])], save)
    # DisplayImage(reduceResolution(LineImages[-1][1],2)[0])


if __name__ == '__main__':
    locationMain = '/media/az/Study/Datasets/ISIDDI_FinalVersion'
    locationLabel = '/media/az/Study/Datasets/ISIDDI_FinalVersion/GroundTruth'
    Data, SegmentedLines, GT, OI,output = '/Data', '/Lines', '/GroundTruth', '/OriginalImages','/Output'
    # targetSet = "/Test"
    targetSet =  "/Validation"
    # targetSet =  "/Training"

    mainLabelPath = locationMain + GT + targetSet
    mainImagePath = locationMain + OI + targetSet
    LinePath = locationMain + SegmentedLines + targetSet
    locationLines = locationMain + output
    DataLoc = locationMain + Data

    dif, counts, ulist = [],[0, 0, 0, 0], np.array([0, 17, 19, 23, 29, 34, 36, 52, 62, 71, 78, 85, 96, 105, 125])
    allFiles = [f for f in listdir(mainLabelPath) if isfile(join(mainLabelPath, f))]
    allLineFiles = np.array([f for f in listdir(locationLines+targetSet) if isfile(join(locationLines+targetSet, f))])
    start, length = 0, 3
    save = True
    # print(len(np.unique(allFiles)))
    # print(allFiles)
    for file in allFiles[6::3]:
    # for file in allFiles[start:start + length]:
    # for file in np.array(allFiles)[ulist]:
        LineSegmentation(file, LinePath, mainImagePath, save)
        # LineMatching(file, counts, mainLabelPath, LinePath)
        # WordMatching(file, LinePath)

    # print(counts)
    # print(dict(collections.Counter(dif)))
    # LineLabeling(allLineFiles,mainLabelPath)
    # WordLabeling(DataLoc)
    exit()
