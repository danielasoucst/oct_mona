# import numpy as np
# from scipy import ndimage as ndi
# from skimage.filters import gabor_kernel
#
# def compute_feats(image, kernels):
#     feats = np.zeros((len(kernels), 2), dtype=np.double)
#     for k, kernel in enumerate(kernels):
#         filtered = ndi.convolve(image, kernel, mode='wrap')
#         feats[k, 0] = filtered.mean()
#         feats[k, 1] = filtered.var()
#     return feats
#
# def getKernels():
#     # prepare filter bank kernels
#     kernels = []
#     for theta in range(4):
#         theta = theta / 4. * np.pi
#         for sigma in (1, 3):
#             for frequency in (0.05, 0.25):
#                 kernel = np.real(gabor_kernel(frequency, theta=theta,
#                                               sigma_x=sigma, sigma_y=sigma))
#                 kernels.append(kernel)
#
# def gabor(frame):
#     kernels = getKernels()
#     ref_feats = np.zeros((1, len(kernels), 2), dtype=np.double)
#     ref_feats[0, :, :] = compute_feats(frame, kernels)

# !/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt

# from skimage import feature
raio = 3
n_points = 8 * raio
eps=1e-7
def build_filters():
    filters = []
    ksize = 20
    # for theta in np.arange(0, np.pi, np.pi / 16):
    #     kern = cv2.getGaborKernel((ksize, ksize), 4.0,theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    #     kern /= 1.5 * kern.sum()
    #     filters.append(kern)
    kern = cv2.getGaborKernel((ksize, ksize), 4.0,90, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    kern /= 1.5 * kern.sum()
    filters.append(kern)
    print (len(filters))
    return filters


def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
    np.maximum(accum, fimg, accum)
    return accum

# def convert_to_LBP(window): #retorna o histograma da imagem ja converida para LBP
#     lbp = feature.local_binary_pattern(window, n_points, raio, method="uniform")
#     (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
#     # normalize the histogram
#     hist = hist.astype("float")
#     hist /= (hist.sum() + eps)
#
#     return hist,lbp


def gabor(frame):
    filters = build_filters()

    res1 = process(frame, filters)
    return res1



# dirTest = '/home/daniela/retina/DADOS/Chiu_IOVS_2011/Automatic versus Manual Study/Group1_Volume2.mat'
# volume = io.load_volume_chiu_iovs_2011(dirTest)
# frame = volume.get_frame(1)
#
# out = cv2.GaussianBlur(frame.data,(5,5),0)
# print ("chamando gabor")
# out = gabor(out)
# _,out2 = convert_to_LBP(out)
#
# plt.figure()
# plt.subplot(121)
# plt.imshow(out, 'gray')
# plt.subplot(122)
# plt.imshow(out2, 'gray')
# plt.show()