import PIL
import cv2
import functools
import numpy as np
from PIL import Image
from skimage.morphology import thin
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


def Plotting(clust,space,reachability,labels_050,labels_200,labels,X):
    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(2, 3)
    ax2 = plt.subplot(G[:,:])
    # ax1 = plt.subplot(G[0, :])
    # ax2 = plt.subplot(G[1, :])
    # ax3 = plt.subplot(G[1, 1])
    # ax4 = plt.subplot(G[1, 2])

    # Reachability plot
    # colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    # for klass in (range(len(clust.labels_))):
    #     Xk = space[labels == klass]
    #     Rk = reachability[labels == klass]
    #     ax1.plot(Xk, Rk, colors[klass%3], alpha=0.3)
    # ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
    # ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
    # ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
    # ax1.set_ylabel('Reachability (epsilon distance)')
    # ax1.set_title('Reachability Plot')

    # OPTICS
    colors = ['g.', 'r.', 'b.']
    pos0,pos1 = 0,1
    for klass in (range(len(np.unique(clust.labels_))-1)):
        Xk = X[clust.labels_ == klass]
        # print(Xk,'\n')
        ax2.plot(Xk[:, pos1], Xk[:, pos0], colors[klass%3], alpha=0.3)
    ax2.plot(X[clust.labels_ == -1, pos1], X[clust.labels_ == -1, pos0], 'k+', alpha=0.1)
    print()
    ax2.set_title('Automatic Clustering\nOPTICS '+str(clust.min_samples))

    # DBSCAN at 0.5
    # colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
    # for klass, color in zip(range(0, 6), colors):
    #     Xk = X[labels_050 == klass]
    #     ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
    # ax3.plot(X[labels_050 == -1, 0], X[labels_050 == -1, 1], 'k+', alpha=0.1)
    # ax3.set_title('Clustering at 0.5 epsilon cut\nDBSCAN')
    #
    # # DBSCAN at 2.
    # colors = ['g.', 'm.', 'y.', 'c.']
    # for klass, color in zip(range(0, 4), colors):
    #     Xk = X[labels_200 == klass]
    #     ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    # ax4.plot(X[labels_200 == -1, 0], X[labels_200 == -1, 1], 'k+', alpha=0.1)
    # ax4.set_title('Clustering at 2.0 epsilon cut\nDBSCAN')

    # plt.xticks(X[clust.labels_ == -1, pos1], X[clust.labels_ == -1, pos0], rotation='vertical')
    # # Pad margins so that markers don't get clipped by the axes
    # plt.margins(0.2)
    # # Tweak spacing to prevent clipping of tick-labels
    # plt.subplots_adjust(bottom=0.15)

    plt.tight_layout()
    plt.show()

def remove_noise_and_smooth(img):
    # img = cv2.imread(file_name, 0)
    # img = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 41)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    or_image = cv2.bitwise_or(img, closing)

    or_image= cv2.invert(or_image)
    kernel = np.ones((1, 1), np.uint8)
    or_image = cv2.erode(or_image, kernel, iterations=1)
    return or_image


def Deskew(image):
    coords = np.column_stack(np.where(image == 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45: angle = -(90 + angle)
    else: angle = -angle

    angle  = angle
    print(angle)

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

def Preprocessing(image):
    blur = cv2.GaussianBlur(image, (1, 1), 0)
    binImage = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # binImage = remove_noise_and_smooth(image)
    processedImage = thin(binImage)
    return  binImage

def TypeAndShape(obj):
    print(type(obj))
    print(np.shape(obj))

def DisplayImage(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def DisplayImages(images):
    for i in images:
        cv2.imshow(str(i[0]), i[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Iterate(image):
    for i in image:
        print(i)
        print('------------------')

def Resize(image):
    basewidth = 1500
    image = image_transpose_exif(image)
    wpercent = (basewidth / float(image.size[0]))
    hsize = int((float(image.size[1]) * float(wpercent)))
    image = image.resize((basewidth, hsize), Image.ANTIALIAS)
    return image

def image_transpose_exif(im):

    exif_orientation_tag = 0x0112
    exif_transpose_sequences = [                   # Val  0th row  0th col
        [],                                        #  0    (reserved)
        [],                                        #  1   top      left
        [Image.FLIP_LEFT_RIGHT],                   #  2   top      right
        [Image.ROTATE_180],                        #  3   bottom   right
        [Image.FLIP_TOP_BOTTOM],                   #  4   bottom   left
        [Image.FLIP_LEFT_RIGHT, Image.ROTATE_90],  #  5   left     top
        [Image.ROTATE_270],                        #  6   right    top
        [Image.FLIP_TOP_BOTTOM, Image.ROTATE_90],  #  7   right    bottom
        [Image.ROTATE_90],                         #  8   left     bottom
    ]

    try:
        seq = exif_transpose_sequences[im._getexif()[exif_orientation_tag]]
    except Exception:
        return im
    else:
        return functools.reduce(type(im).transpose, seq, im)

def PreprocessingAdaptive(image,param=105):
    image =  cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, param, 12)
    return image

if __name__ == '__main__':

    # image = cv2.imread("1.jpg", 0)
    # image = cv2.imread("2.jpg", 0)
    # image = cv2.imread("3.jpg", 0)
    images = []

    image = Image.open('3.jpg')
    image = Resize(image)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # DisplayImage(image)

    # image.save('resized_image.jpg')

    # mean_c = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 12)
    # gaus = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 12)

    # for itr in range(15,20):
    # for itr in range(79,129,2):
        # images.append((itr,cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, itr)))
        # images.append((itr,cv2.adaptiveThreshold(image.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 105, 15)))

    image = cv2.adaptiveThreshold(image.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,105, 15)
    # image = remove_noise_and_smooth(image)
    DisplayImage(image)

    # DisplayImages(images)
    exit()

    # gray = cv2.bitwise_not(image)
    # thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # rotated= Deskew(mean_c)
    # rotated = Preprocessing(gray)

    # thresh = cv2.adaptiveThreshold(image.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 41)[1]
    # cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)




