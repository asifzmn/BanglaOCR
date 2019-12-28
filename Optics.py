import os
from os import path

import cv2
from sklearn import preprocessing
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import numpy as np
from ImagePreparation import Preprocessing, PrepareImage, DisplayImage, Opening, reduceResolution, Saving,\
    DisplayImages, PrepareImage1


def subFoldering():
    subfolders = [f.path for f in os.scandir('/home/az/PycharmProjects/BanglaOCR/WordSeg') if f.is_dir()]
    for subfolder in subfolders[:1]:
        files =sorted([f for f in os.listdir(subfolder)])

        src1 = path.realpath(subfolder)
        old_file = os.path.join(src1, "0.png")
        new_file = os.path.join(src1, "0.png")
        os.rename(old_file, new_file)

        # for file in files:
        #     old_file = os.path.join(src1, "a.txt")
        #     new_file = os.path.join(src1, "b.kml")
        #     os.rename(old_file, new_file)

def Opening1(image):
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((2, 4), np.uint8), iterations=3)

def Clustering(imageMain,minSamples=40, xI=.1, minClusterSize=.01,scaling = 15,p=2.0,word = 1):
    image = np.copy(imageMain)
    imageMain = cv2.cvtColor(imageMain,cv2.COLOR_GRAY2RGB)
    image,ratio = reduceResolution(image, uniformWidth=True, word=word)
    # DisplayImage(image)
    image = Opening(image,word)
    # DisplayImage(image)
    image = Preprocessing(image)
    # DisplayImage(image)
    X = np.multiply(np.argwhere(image == 0), [scaling,1])

    # np.random.seed(0)
    # n_points_per_cluster = 250
    #
    # C1 = [-5, -2] + .8 * np.random.randn(n_points_per_cluster, 2)
    # C2 = [4, -1] + .1 * np.random.randn(n_points_per_cluster, 2)
    # C3 = [1, -2] + .2 * np.random.randn(n_points_per_cluster, 2)
    # C4 = [-2, 3] + .3 * np.random.randn(n_points_per_cluster, 2)
    # C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
    # C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
    # if X is None: X = np.vstack((C1, C2, C3, C4, C5, C6))

    # labels_050 = cluster_optics_dbscan(reachability=clust.reachability_,
    #                                    core_distances=clust.core_distances_,
    #                                    ordering=clust.ordering_, eps=0.5)
    # labels_200 = cluster_optics_dbscan(reachability=clust.reachability_,
    #                                    core_distances=clust.core_distances_,
    #                                    ordering=clust.ordering_, eps=2)
    imageColor = np.copy(imageMain)

    clust = OPTICS(min_samples=minSamples, xi=xI, min_cluster_size=minClusterSize, p=p)

    clust.fit(X)

    space = np.arange(len(X))
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]
    # print(len(np.unique(labels)) - 1)

    LineImages = []
    # colorShade = [ [b,g,r] for r,g,b in [(147,112,219),(32,178,170),(240,230,140)] ]
    # colorShade = [ [b,g,r] for r,g,b in [(255,105,180),(147,112,219),(30,144,255)] ]
    colorShade = [[b, g, r] for r, g, b in [(255, 105, 180), (153, 50, 204), (0, 191, 255)]]
    missfire = 0

    for i, x in enumerate(range(len(np.unique(clust.labels_)) - 1)):

        X0 = np.ceil(np.multiply(((X[labels == x].astype('float'))), [1.0 / float(scaling), 1.0])) * ratio
        upperLeft = np.array(np.floor(X0.min(axis=0))).astype('int')
        lowerRight = np.array(np.ceil(X0.max(axis=0))).astype('int')
        # print(upperLeft,lowerRight)

        # X0 = X[labels == x]
        # upperLeft = np.round(np.multiply((((X0.min(axis=0).astype('float')))) * ratio, [1.0 / float(scaling), 1.0])).astype('int')
        # lowerRight =np.round(np.multiply((((X0.max(axis=0).astype('float')))) * ratio, [1.0 / float(scaling), 1.0])).astype('int')

        coordinates = np.array((upperLeft, lowerRight))
        if coordinates[0][0] >= coordinates[1][0] or coordinates[0][1] >= coordinates[1][1]:
            missfire += 1
            continue
        segment = imageMain[coordinates[0][0]:coordinates[1][0], coordinates[0][1]:coordinates[1][1]]
        segmentC = imageColor[coordinates[0][0]:coordinates[1][0], coordinates[0][1]:coordinates[1][1]]
        LineImages.append((str(i - missfire), segment))
        # segment[segment==0]=greyShade[i%2]

        for r in range(np.shape(segmentC)[0]):
            for c in range(np.shape(segmentC)[1]):
                if np.array_equal(segmentC[r][c], [0, 0, 0]):
                    segmentC[r][c] = colorShade[i % 3]

    # Plotting(clust, space, reachability, labels_050, labels_200, labels, X)
    LineImages.append(('color', imageColor))
    return LineImages



if __name__ == '__main__':
    samples = ['sample.jpg','sample1.png','sample2.jpg','sample3.png','sample4.jpg','sample5.png','sample6.png','sample7.png','sample8.png','sample9.png','sample10.png','sample11.png']
    imageCollection = ['images.png','1.jpg','2.jpg','3.jpg','sampleHindi.png','test.jpeg','camera.jpg','crop1.png']
    lineCollection = ['DhyanOSakti_pg111.txt','Gospelofmatthewi_pg032.txt','ManahShiksha_pg039.txt','OnlyWayTobeSaved_pg015.txt','SankhaDarsanam_pg096.txt']
    fileName = 'Images/' + imageCollection[0]
    # fileName = '/media/az/Study/Datasets/ISIDDI_FinalVersion/OriginalImages/Test/DhyanOSakti_pg005.jpeg'
    # fileName = '/media/az/Study/Datasets/ISIDDI_FinalVersion/OriginalImages/Test/Bardidi_pg068.jpeg'

    image = PrepareImage(fileName)
    # DisplayImage(image)
    # LineImages = Clustering(image,minSamples=20, xI=.2, minClusterSize=.01,scaling = 5)
    LineImages = Clustering(image,minSamples=33, xI=.15, minClusterSize=.0075,scaling = 15)

    for i in range(len(LineImages)-1):
        image = np.transpose(PrepareImage1(LineImages[i][1]))
        lineImages = Clustering(image, minSamples=15, xI=.03, minClusterSize=.025, scaling=1, word=0, p=1.5)
        # lineImages = Clustering(image, minSamples=33, xI=.15, minClusterSize=.005, scaling=3, word=0, p=1.15)
        lineImages = [(a, np.transpose(b, axes=[1, 0, 2])) for a, b in lineImages[-1::]]
        DisplayImages(lineImages)


    # Saving(fileName,Lines)

    # a = np.array([[1,4],[3,6]])
    # b = preprocessing.normalize(a, norm='l2')
    # print(b)
    # pos = np.where(b==b[0][0])
    # print(a[pos[0],pos[1]])


    # print('Tuple of arrays returned : ', result)
    # for i in range(0,15,3):
    #     for j in range(5):
    #         fileName = 'WordSeg/' + lineCollection[j] + '/'+str(i)+'.png'
    #         image = np.transpose(PrepareImage(fileName))
    #         # LineImages = Clustering(image, minSamples=3, xI=.5, minClusterSize=.05, scaling=3,word = 0,p=1.15)
    #         lineImages = Clustering(image, minSamples=15, xI=.03, minClusterSize=.075, scaling=1, word=0, p=1.5)
    #         lineImages = [(a, np.transpose(b, axes=[1, 0, 2])) for a, b in lineImages[-1::]]
    #         # DisplayImages(lineImages)


