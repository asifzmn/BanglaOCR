import os
import cv2
from skimage.filters import gaussian, threshold_otsu
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line, rotate
import numpy as np

def Saving(targetDir, segmentedParts,save):
    if not save:return
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
    for image in segmentedParts:
            cv2.imwrite(targetDir + '/' + image[0] + '.png', image[1])

def DisplayImage(processedImage):
    if processedImage is None : return
    winname = "Image"
    cv2.namedWindow(winname)  # Create a named window
    cv2.moveWindow(winname, 1000, 100)
    cv2.imshow(winname, processedImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def DisplayImages(images):
    for i in images:cv2.imshow(i[0], i[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def deskew(image,blurring = 3):
    # DisplayImage(image)
    # threshold to get rid of extraneous noise
    thresh = threshold_otsu(image)
    normalize = image > thresh

    # gaussian blur
    blur = gaussian(normalize,blurring)

    # canny edges in scikit-image
    edges = canny(blur)
    # DisplayImage((edges.astype(np.uint8)*255))
    # hough lines
    hough_lines = probabilistic_hough_line(edges)

    # hough lines returns a list of points, in the form ((x1, y1), (x2, y2))
    # representing line segments. the first step is to calculate the slopes of
    # these lines from their paired point values
    slopes = [(y2 - y1) / (x2 - x1) if (x2 - x1) else 0 for (x1, y1), (x2, y2) in hough_lines]

    # it just so happens that this slope is also y where y = tan(theta), the angle
    # in a circle by which the line is offset
    rad_angles = [np.arctan(x) for x in slopes]

    # and we change to degrees for the rotation
    deg_angles = [np.degrees(x) for x in rad_angles]

    # which of these degree values is most common?
    histo = np.histogram(deg_angles, bins=180)

    # correcting for 'sideways' alignments
    rotation_number = histo[1][np.argmax(histo[0])]

    if rotation_number > 45:rotation_number = -(90 - rotation_number)
    elif rotation_number < -45:rotation_number = 90 - abs(rotation_number)

    # print(rotation_number)
    # DisplayImage(rotate(image, rotation_number,  resize=True,cval=1))
    return rotate(image, rotation_number,  resize=True,cval=1)

def Opening(image,word):
    setting = [[(2,3),1],[(1,3),3]]
    if word == 1:return cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones(setting[word][0], np.uint8), iterations=setting[word][1])
    else:return cv2.erode(image, np.ones(setting[word][0], np.uint8), iterations=setting[word][1])

def reduceResolution(image, ratio=1, width=200, uniformWidth=False, word=1):
    # if dim == 0: uniformWidth = False
    if not uniformWidth:return cv2.resize(image, (0, 0), fx=1/ratio, fy=1/ratio),ratio
    ratio = np.shape(image)[word] / width
    return cv2.resize(image, (0, 0), fx=1/ratio, fy=1/ratio),ratio

def Preprocessing(image):
    image = cv2.GaussianBlur(image, (3, 3),1)
    image =  cv2.threshold(image, 0, 255,  cv2.THRESH_OTSU)[1]
    return image

def PrepareImage(path):
    rotatedImage = deskew(Preprocessing(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)))
    return Preprocessing(cv2.normalize(rotatedImage, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8))

def PrepareImage1(image):
    rotatedImage = deskew(Preprocessing(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))
    return Preprocessing(cv2.normalize(rotatedImage, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8))

if __name__ == '__main__':
    samples = ['sample.jpg','sample1.png','sample2.jpg','sample3.png','sample4.jpg','sample5.png','sample6.png','sample7.png','sample8.png','sample9.png','sample10.png','sample11.png']
    imageCollection = ['images.png','1.jpg','2.jpg','3.jpg','sampleHindi.png','test.jpeg','camera.jpg','crop1.png']
    fileName = 'Images/' + imageCollection[0]

    image = reduceResolution(PrepareImage("/media/az/Study/Datasets/ISIDDI_FinalVersion/OriginalImages/Training/MohonerAgatobas_pg005.jpeg"),1)[0]
    # image = PrepareImage(fileName)
    DisplayImage(image)
    exit()