from skimage.feature import hog
import numpy as np
import cv2
import gabor
import glcm

from skimage.viewer import ImageViewer
from skimage import  data, color, exposure
from matplotlib import pyplot as plt

def apply_hog(image):
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(4, 4),
                        cells_per_block=(2, 2), visualise=True)

    # generate Gaussian pyramid for A
    # G = np.copy(image)
    # gpA = [G]
    # for i in xrange(3):
    #     G = cv2.pyrDown(G)
    #     gpA.append(G)
    #
    # for img in gpA:
    #     viewer = ImageViewer(img)
    #     viewer.show()
    # Rescale histogram for better display
    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(91,301))


    print ('fd',fd.shape,hog_image.shape)
    return fd


def hog2(image):
    winSize = (4, 4)
    blockSize = (2, 2)
    blockStride = (1, 1)
    cellSize = (2, 2)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 4
    signedGradients = True

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradients)

    descriptor = hog.compute(image)
    print('tam:', descriptor.shape)
    return descriptor

def apply_sift(image):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(image,None)
    img = np.copy(image)
    cv2.drawKeypoints(image, kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kp2, des = sift.detectAndCompute(img,None)

   # print('key:',des.shape)



    return des


def apply_BOW(lstfeatures):
    dictionarySize = 20

    BOW = cv2.BOWKMeansTrainer(dictionarySize)

    for p in lstfeatures:
      # print p.shape
        BOW.add(p)

    # dictionary created
    dictionary = BOW.cluster()
    print ('dic',dictionary.shape)
    return dictionary

def getHistogram(window):
    eps = 1e-7
    (hist, _) = np.histogram(window.ravel(), bins=np.arange(0,256), range=(0, 256))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist


def apply_gabor(imagem):

    imagem_gabor = gabor.gabor(imagem)
    hist = getHistogram(imagem_gabor)
    return hist

def apply_glcm(imagem):

    estatisticas_glcm = glcm.glcm_features(imagem)
    return estatisticas_glcm