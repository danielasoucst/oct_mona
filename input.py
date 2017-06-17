# coding: utf-8
import os
import cv2
import heapq
import numpy as np
from matplotlib import pyplot as plt

def load_image(dir):
    lstImages = []
    for file in os.listdir(dir):
        if file.endswith(".tif"):
            volumeName = os.path.join(dir, file)
            print (volumeName)
            img = cv2.imread(volumeName,0)
            lstImages.append(img)
            # plt.figure()
            # plt.subplot(121)
            # plt.imshow(img, 'gray')
            # plt.show()

    return lstImages
def apply_filter(image,metodo):
    print(image.shape)
    if(metodo=='gauss'):
        out = cv2.GaussianBlur(image, (1, 3), 0)
        return out
    return None

def find_polynomial(lst_x,lst_y,largura):
    print('x=: ',lst_x)
    print('y=: ', lst_y)
    print('x,y',len(lst_x),len(lst_y))

    z = np.polyfit(lst_x, lst_y, 2)
    f = np.poly1d(z)

    x_new = np.linspace(lst_x[0], lst_x[-1], largura)
    y_new = f(x_new)

    return y_new.tolist()

def flat_image(image):
    print image.shape
    pilot_rpe = []
    list_x = []

    for c in range(0,image.shape[1]):
        column = image[0:image.shape[0],c]
        column_list = column.tolist()
        # if 255 in column_list:
        #     column_list = list(filter(lambda x: x!= 255, column_list))
        # thres = np.sum(heapq.nlargest(5, xrange(len(column_list)), column_list.__getitem__))/5
        for i in range(len(column_list)-1,0,-1):
           if( column_list[i] > 100 and column_list[i] != 255):
                list_x.append(c)
                pilot_rpe.append(i)
                break

    if (len(pilot_rpe) > 0):
        ele = max(pilot_rpe)
        limit = [ele for x in xrange(0, image.shape[1])]

    poly = find_polynomial(list_x, pilot_rpe,image.shape[1])
    print pilot_rpe
    print(limit)
    return pilot_rpe,limit,poly
    # column = image[0:image.shape[0], 250]
    # print column




lstImagens = load_image('./Publication_Dataset/AMD2/TIFFs/8bitTIFFs/')

frame_denoise = apply_filter(lstImagens[1],'gauss')
line,line2,poly = flat_image(frame_denoise)
from skimage import feature

edges2 = feature.canny(frame_denoise, sigma=2)
print len(line)
plt.figure()
plt.subplot(121)
plt.imshow(lstImagens[1], 'gray')
plt.subplot(122)
plt.imshow(frame_denoise, 'gray')
plt.plot(line, '.', lw=1)
# plt.plot(line2, 'ro', lw=1)
plt.plot(poly, 'ro', lw=1)
plt.show()

