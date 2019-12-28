import os
import numpy as np
import collections
import matplotlib.pyplot as plt
import cv2
from itertools import groupby
from skimage.util import invert

black, darkgray, gray, lightgray, white = 0, 64, 128, 192, 255
# darkgray = 32
color = [lightgray,darkgray]
showHeader = not True
showMids =  not True
showUpper = not True
displayEach = not True
save = not True
Threshold = .5


def printWithNumber(item):
    for i in range(len(item)):
        print(i, item[i])
    print()


def custom_sort(t):
    return t[0]


def Saving(fileName, segmentedParts):
    if save:
        if not os.path.exists('Output/' + fileName):
            os.makedirs('Output/' + fileName)

        for line in segmentedParts:
            for image in line:
                cv2.imwrite('Output/' + fileName + '/' + image[0] + '.png', image[1])
        return True
    return False


def plot_bar_x(Y):
    N = len(Y)
    X = range(N)

    plt.bar(X, Y)
    plt.xlabel('X', fontsize=5)
    plt.ylabel('Y', fontsize=5)
    plt.xticks(Y, X, fontsize=5, rotation=30)
    plt.title('Bar Chart')
    plt.show()


def remove_noise_and_smooth(img):
    # img = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 41)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    or_image = cv2.bitwise_or(img, closing)

    or_image = invert(or_image)
    kernel = np.ones((1, 1), np.uint8)
    or_image = cv2.erode(or_image, kernel, iterations=1)
    return or_image


def Preprocessing(image):
    blur = cv2.GaussianBlur(image, (1, 1), 0)
    binImage = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # binImage = remove_noise_and_smooth(image)
    # processedImage = thin(binImage)
    return binImage


def DisplayImage(processedImage):
    if processedImage is None or len(processedImage): return
    cv2.imshow("Image", processedImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def DisplayImages(images):
    for i in images:
        cv2.imshow(i[0], i[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def BaseLineDetection(processedImage, lines, headerRange):

    heightList = []
    baseline = []

    for line in lines:
        for column in range(len(processedImage[0])):

            for row in range(line[1] - 1, line[0], -1):
                if (row == headerRange[lines.index(line)][0]):
                    continue
                if (processedImage[row][column] == 0 and row - (headerRange[lines.index(line)][0]) != 0):
                    heightList.append(row - headerRange[lines.index(line)][0])
                    break

    # heightList = [x for x in heightList if x > 2 * max(heightList) / 3]
    heightList = [x for x in heightList if x > 1*max(heightList)/2]
    # print(heightList)

    freqs = groupby(collections.Counter(heightList).most_common(), lambda x: x[1])
    # charheight = statistics.mode(heightList)
    charheight = max([val for val, count in next(freqs)[1]])

    for i in range(len(lines)):
        baseline.append(headerRange[i][0] + charheight)

    # processedImage[headerRange[i][0] + charheight - 1 : headerRange[i][0] + charheight + 1] = lightgray
    return baseline



def HorizontalProjectionProfile(processedImage, lines):
    headerPositions = []
    headerRange = []
    for line in lines:
        pixelList = []
        for row in range(line[0], line[1]):
            counter = collections.Counter(processedImage[row])
            pixelList.append(counter[0])

        pos = pixelList.index(max(pixelList)) + line[0]
        headerPositions.append(pos)
        # processedImage[pos - 1:pos + 1] = darkgray


        headerRange.append((HeaderRange(processedImage, line, pos, -1), HeaderRange(processedImage, line, pos, 1)))

    return headerPositions, headerRange


def HeaderRange(processedImage, line, headerPosition, direction):
    Threshold = .5

    currentPos = headerPosition
    thresholdvVal = collections.Counter(processedImage[headerPosition])[0] * Threshold

    while (collections.Counter(processedImage[currentPos + direction])[0] > thresholdvVal and (
            headerPosition > line[0] and headerPosition < line[1])):
        currentPos = currentPos + direction

    if showHeader:
        if direction==1:
            processedImage[currentPos]=lightgray
        else:
            processedImage[currentPos] = darkgray

    return currentPos

    # img = cv2.imread('images.png', 0)
    # slicing
    # img[:,:10] = 255

    # masking
    # mask = img < 87
    # img[mask] = 255

    # fancy indexing
    # inds_r = np.arange(len(img))
    # inds_c = 4 * inds_r % len(img)
    # img[inds_r, inds_c] = 0


def SeparationAndFormatting(processedImage, lines, headerRange, rowColumnList, connections):
    count = 0

    segmentedImages = []
    for i in range(len(lines)):
        segmentedLineImages = []
        end = lines[i][1]
        upperStart = lines[i][0]
        upperEnd = headerRange[i][0]
        keys = list(connections[i].keys())

        deletePos = []
        # print(rowColumnList[i])

        for mid in range(len(rowColumnList[i]) - 1):
            cutImage = processedImage[upperEnd:end, rowColumnList[i][mid]:rowColumnList[i][mid + 1]]

            if mid in keys:

                cutImageUpper = processedImage[upperStart:upperEnd, connections[i][mid][0]:connections[i][mid][1]]

                l1 = rowColumnList[i][mid]
                l2 = connections[i][mid][0]

                r1 = rowColumnList[i][mid + 1]
                r2 = connections[i][mid][1]

                CCcount = 1000

                if (connections[i][mid][2] != 0):
                    CCcount = ConnectedComponentCount(processedImage[upperStart:end, min(l1, l2):max(r1, r2)])

                if (CCcount == 1 and connections[i][mid][2] != 0):
                    cutImageOther = processedImage[upperStart:end, min(l1, l2):max(r1, r2)]
                    targetImage = (str(count), cutImageOther)
                    deletePos.append(len(segmentedLineImages) + connections[i][mid][2])

                else:
                    midAndUpper = 255 * np.ones(shape=[end - upperStart, max(r1, r2) - min(l1, l2)], dtype=np.uint8)

                    midAndUpper[np.shape(midAndUpper)[0] - (end - upperEnd):,
                    rowColumnList[i][mid] - min(l1, l2):rowColumnList[i][mid + 1] - min(l1, l2)] = cutImage
                    midAndUpper[:upperEnd - upperStart,
                    connections[i][mid][0] - min(l1, l2):connections[i][mid][1] - min(l1, l2)] = cutImageUpper

                    coord = (upperStart, upperEnd, connections[i][mid][0], connections[i][mid][1])
                    # title = str(coord[0]) + ' ' + str(coord[1]) + ' ' + str(coord[2]) + ' ' + str(coord[3])
                    targetImage = (str(count), midAndUpper)

            else:
                # coord = (start, end, rowColumnList[i][mid], rowColumnList[i][mid + 1])
                # title = str(coord[0]) + ' ' + str(coord[1]) + ' ' + str(coord[2]) + ' ' + str(coord[3])
                targetImage = (str(count), cutImage)

            count += 1
            segmentedLineImages.append(targetImage)
            if displayEach: DisplayImage(targetImage[1])

        for deleteItem in deletePos[::-1]:
            segmentedLineImages.pop(deleteItem)

        segmentedImages.append(segmentedLineImages)
    return segmentedImages


def UpperLigature(rowColumnList, lines, headerRange):
    connections = []

    for i in range(len(lines)):
        upperPart = (image[lines[i][0]:headerRange[i][0]])
        t_matrix = np.transpose(upperPart)
        positions = LineSeparation(t_matrix, 0)

        connection = {}
        current = 0
        for left, right in positions:
            while (left > rowColumnList[i][current][1] and current < len(rowColumnList[i])):
                current += 1
            if (right <= rowColumnList[i][current][1]):
                connection[current] = ([left, right])
            else:
                leftPart = rowColumnList[i][current][1] - left
                rightPart = right - rowColumnList[i][current][1]

                if (leftPart < rightPart):
                    connection[current] = ([left, right])
                else:
                    connection[current + 1] = ([left, right])

        connections.append(connection)

    return connections


def UpperLigature1(image,rowColumnList, lines, headerRange):
    connections = []

    for i in range(len(lines)):
        upperPart = (image[lines[i][0]:headerRange[i][0]])
        t_matrix = np.transpose(upperPart)
        positions = LineSeparation(t_matrix, 2)
        # boundaryList.append(positions)

        connection = {}
        current = 0
        for left, right in positions:
            # print(len(rowColumnList[i]))
            while ((left > rowColumnList[i][current]) and (current < len(rowColumnList[i]) - 1)):
                current += 1
                # print(current,len(rowColumnList[i]))
            if (right <= rowColumnList[i][current]):
                # connection.append([current,left,right])
                connection[current - 1] = ([left, right, 0])
            else:
                leftPart = rowColumnList[i][current] - left
                rightPart = right - rowColumnList[i][current]

                if (leftPart < rightPart):
                    connection[current - 1] = ([left, right, 1])
                else:
                    connection[current] = ([left, right, -1])

        connections.append(connection)
        # print(connections)

    return connections


def LineSeparation(imageCut, adjuster):
    linesScopes = []
    lineSelect = False
    lineStart = 0
    imageCopy = np.copy(imageCut)

    for row in range(len(imageCut)):
        # print(len(set(imageCut[row])))

        if not (len(set(imageCut[row])) == 1) and not lineSelect:
            # if(adjuster==1): imageCopy[row - 1:row + 1] = lightgray
            # if(adjuster==2): imageCut[row - 1:row ] = darkgray
            lineSelect = True
            lineStart = row

        if (len(set(imageCut[row])) == 1) and lineSelect:  # all white
            # if(adjuster==1): imageCopy[row - 1:row + 1] = lightgray
            # if(adjuster==2): imageCut[row - 1:row ] = darkgray
            lineSelect = False
            lineEnd = row
            linesScopes.append((lineStart, lineEnd))

    # if(adjuster==1 or adjuster==2):DisplayImage(imageCut)
    return linesScopes


def WordAndCharacterSeparation(imageCut, lineIndex, widthsOfLines):
    leftSelect = False
    rightSelect = False

    leftPos = 0
    rightPos = 0
    mids = [widthsOfLines[lineIndex][0]]

    for row in range(widthsOfLines[lineIndex][0], widthsOfLines[lineIndex][1]):

        if (set(imageCut[row])) == {255} and not leftSelect:  # all white
            # imageCut[row - 1:row + 1] = gray
            # print('not left select ',row)
            leftSelect = True
            leftPos = row

        if ((set(imageCut[row])) == {0} or (set(imageCut[row])) == {255, 0}) and leftSelect:
            # imageCut[row - 1:row + 1] = gray
            # print('leftselect ',row)
            rightSelect = True
            rightPos = row - 1

        if leftSelect and rightSelect:
            midPos = int((rightPos + leftPos) / 2)
            mids.append(midPos)
            if showMids:imageCut[midPos - 1:midPos + 1] = darkgray
            else:imageCut[midPos - 1:midPos + 1] = white
            leftSelect = False
            rightSelect = False

    mids.append(widthsOfLines[lineIndex][1])
    # mids.insert(mids.index(165)+1, 174)
    # mids.insert(mids.index(431) + 1, 441)
    mids = list(dict.fromkeys(mids))
    return mids


def WordAndCharacterSeparation1(imageCut, lineIndex):
    # DisplayImage(imageCut)
    leftSelect = False
    rightSelect = False
    nothingOnLeft = True

    leftPos = 0
    rightPos = 0
    mids = []
    ignore = []

    for row in range(len(imageCut)):

        # print(row, leftSelect,rightSelect,nothingOnLeft)
        # print(set(imageCut[row]))
        # if(headerPositions[lineIndex]==255):
        # if(processedImage[headerPositions[lineIndex]][row]==255) and (set(imageCut[row]))=={255} :
        #     # print('blnak ',row)
        #     if leftSelect:
        #         mids.append(int(row-1))
        #         ignore.append(len(mids)-1)
        #         leftSelect=False
        #     nothingOnLeft=True
        #     continue

        if (set(imageCut[row])) == {255} and not leftSelect:  # all white
            # imageCut[row - 1:row + 1] = gray
            # print('not left select ',row)
            leftSelect = True
            leftPos = row

        if ((set(imageCut[row])) == {0} or (set(imageCut[row])) == {255, 0}) and leftSelect:
            # imageCut[row - 1:row + 1] = gray
            # print('leftselect ',row)
            rightSelect = True
            rightPos = row - 1

        if leftSelect and rightSelect:
            # mid = rightPos-leftPos
            if (nothingOnLeft):
                # print('nothing ',row)
                # imageCut[int(leftPos) - 1:int(leftPos) + 1] = white
                mids.append(int(leftPos))
                nothingOnLeft = False
            else:
                # print(leftPos,rightPos)
                # imageCut[int((rightPos + leftPos) / 2) - 1:int((rightPos + leftPos) / 2) + 1] = white
                mids.append(int((rightPos + leftPos) / 2))
            leftSelect = False
            rightSelect = False


    # print(mids)
    return mids


def CandidateColumns(processedImage, headerRange, lines, baseLines, widthsOfLines):
    candidateColumnsList = []

    for i in range(len(lines)):
        lineWithoutHeader = (processedImage[headerRange[i][1] + 1:baseLines[i] + 1])
        t_matrix = np.transpose(lineWithoutHeader)
        positions = WordAndCharacterSeparation(t_matrix, i, widthsOfLines)


        for position in positions:
            processedImage[headerRange[i][0]:baseLines[i] + 1, position] = white

        candidateColumnsList.append(positions)
    return candidateColumnsList


def ConnectedComponentCount(image):
    # DisplayImage(image)
    shape = np.shape(image)
    count = 0
    visited = False * np.ones(shape=[shape[0], shape[1]], dtype=np.uint8)

    for i in range(shape[0]):
        for j in range(shape[1]):
            if (visited[i][j] == False and image[i][j] == black):
                count += 1
                image, visited = DFS(image, visited, i, j)

    return count


def DFS(image, visited, i, j):
    # print(i,j)
    visited[i][j] = True
    directions = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1]]

    for direction in directions:
        x, y = i + direction[0], j + direction[1]
        if (x < 0 or y < 0 or x == np.shape(image)[0] or y == np.shape(image)[1]): continue
        if (visited[x][y] == True or image[x][y] != black): continue
        image, visited = DFS(image, visited, x, y)
    return image, visited


def sidePos(image):
    for row in range(len(image)):
        if (len(set(image[row])) == 2):
            return row


def WidthDetection(processedImage, lines):
    widthsOfLines = []
    for line in lines:
        startPos = sidePos(processedImage[line[0]:line[1]].T)
        endPos = np.shape(processedImage)[1] - sidePos(processedImage[line[0]:line[1]].T[::-1])
        widthsOfLines.append((startPos, endPos))

    return widthsOfLines


if __name__ == '__main__':
    directory = 'Images/'
    # fileName = 'sample11.png'
    fileName = 'images.png'
    # fileName = 'sampleHindi.png'

    image = cv2.imread(directory + fileName, 0)
    image = Preprocessing(image)

    lines = LineSeparation(image, 0)
    widthsOfLines = WidthDetection(image, lines)
    headerPositions, headerRange = HorizontalProjectionProfile(image, lines)
    baseLines = BaseLineDetection(image, lines, headerRange)
    rowColumnList = CandidateColumns(image, headerRange, lines, baseLines, widthsOfLines)
    connections = UpperLigature1(image,rowColumnList, lines, headerRange)
    segmentedParts = SeparationAndFormatting(image, lines, headerRange, rowColumnList, connections)
    # saveSuccess = Saving(fileName, segmentedParts)
    DisplayImage(image)
    exit()

    # print(rowColumnList)
    # print(processedImage)
    # print(lines)
    # print(headerPositions)
    # print(type(processedImage))
    # print((processedImage))
    # print(connections)

    # image = remove_noise_and_smooth(cv2.imread('images.png', 0))
    # image = remove_noise_and_smooth(cv2.imread('sample.jpg', 0))
    # image = remove_noise_and_smooth(cv2.imread('test.jpeg', 0))
    # image = remove_noise_and_smooth(cv2.imread('scr1.png', 0))
    # image = remove_noise_and_smooth(cv2.imread('crop1.png', 0))
    # image = remove_noise_and_smooth(cv2.imread('camera.jpg', 0))
    # image = invert(cv2.imread('scr1.png', 0))
    # image = invert(cv2.imread('sample.jpg', 0))
    # image = invert(cv2.imread('test.jpeg', 0))
    # image = invert(cv2.imread('crop1.png', 0))processedImage
    # image = invert(cv2.imread('camera.jpg', 0))

    # plt.imshow(processedImage)
    # plt.show()
