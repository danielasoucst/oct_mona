# coding: utf-8
from matplotlib import pyplot as plt
import input,flatten,cropping,features_extraction,arffGenerator
import pickle

def gerarHOGFeatures(sujeito):
    lstImagens = input.load_image('./Publication_Dataset/'+sujeito+'/TIFFs/8bitTIFFs/')

    lstHOGFeatures=[]

    for imagem in lstImagens:
        frame_denoise = input.apply_filter(imagem,'gauss')
        bdValue, new = flatten.flat_image(frame_denoise)
        crop = cropping.croppy_mona(new,bdValue)
        lstHOGFeatures.append(features_extraction.apply_hog(crop).tolist())
        # features_extraction.apply_sift(crop)
        # lstSIFTFeatures.append(features_extraction.apply_sift(crop))

    print('hog',len(lstHOGFeatures),len(lstHOGFeatures[0]))
    # dictionary = features_extraction.apply_BOW(lstSIFTFeatures)
    fileObject = open('./hog_features/'+sujeito, 'wb')
    if (fileObject != None):
        print('salvando...')
        pickle.dump(lstHOGFeatures, fileObject)
        fileObject.close


def geraGABORFeatures(sujeito):
    lstImagens = input.load_image('./Publication_Dataset/'+sujeito+'/TIFFs/8bitTIFFs/')

    lstGABORFeatures=[]

    for imagem in lstImagens:
        frame_denoise = input.apply_filter(imagem,'gauss')
        bdValue, new = flatten.flat_image(frame_denoise)
        crop = cropping.croppy_mona(new,bdValue)
        lstGABORFeatures.append(features_extraction.apply_gabor(crop).tolist())
        # features_extraction.apply_sift(crop)
        # lstSIFTFeatures.append(features_extraction.apply_sift(crop))

    print('gabor',len(lstGABORFeatures),len(lstGABORFeatures[0]))
    # dictionary = features_extraction.apply_BOW(lstSIFTFeatures)
    fileObject = open('./gabor_features/'+sujeito, 'wb')
    if (fileObject != None):
        print('salvando...')
        pickle.dump(lstGABORFeatures, fileObject)
        fileObject.close

def geraGLCMFeatures(sujeito):
    lstImagens = input.load_image('./Publication_Dataset/'+sujeito+'/TIFFs/8bitTIFFs/')

    lstGLCMFeatures=[]

    for imagem in lstImagens:
        frame_denoise = input.apply_filter(imagem,'gauss')
        bdValue, new = flatten.flat_image(frame_denoise)
        crop = cropping.croppy_mona(new,bdValue)
        lstGLCMFeatures.append(features_extraction.apply_glcm(crop))
        # features_extraction.apply_sift(crop)
        # lstSIFTFeatures.append(features_extraction.apply_sift(crop))

    print('glcm',len(lstGLCMFeatures),len(lstGLCMFeatures[0]))
    # dictionary = features_extraction.apply_BOW(lstSIFTFeatures)
    fileObject = open('./glcm_features/'+sujeito, 'wb')
    if (fileObject != None):
        print('salvando...')
        pickle.dump(lstGLCMFeatures, fileObject)
        fileObject.close

def geraLBPFeatures(sujeito):
    lstImagens = input.load_image('./Publication_Dataset/'+sujeito+'/TIFFs/8bitTIFFs/')

    lstLBPFeatures=[]

    for imagem in lstImagens:
        frame_denoise = input.apply_filter(imagem,'gauss')
        bdValue, new = flatten.flat_image(frame_denoise)
        crop = cropping.croppy_mona(new,bdValue)
        lstLBPFeatures.append(features_extraction.apply_lbp(crop).tolist())
        # features_extraction.apply_sift(crop)
        # lstSIFTFeatures.append(features_extraction.apply_sift(crop))

    print('lbp',len(lstLBPFeatures),len(lstLBPFeatures[0]))
    # dictionary = features_extraction.apply_BOW(lstSIFTFeatures)
    fileObject = open('./lbp_features/'+sujeito, 'wb')
    if (fileObject != None):
        print('salvando...')
        pickle.dump(lstLBPFeatures, fileObject)
        fileObject.close



def gerarTrain(files,classes,fileName,strClasses,dir):

    allFeatures = []
    allLabels = []
    for i in range(0,len(files)):
        fileObject = open(files[i], 'rb')
        features = pickle.load(fileObject)
        vectLabels = [classes[i] for j in range(0, len(features))]
        allFeatures += features
        allLabels += vectLabels

    arffGenerator.createArffFile(dir + fileName,allFeatures,allLabels,strClasses,len(allFeatures[0]))


def testarEnquadramento():
    lstImagens = input.load_image('./Publication_Dataset/NORMAL1/TIFFs/8bitTIFFs/')


    for imagem in lstImagens:
        frame_denoise = input.apply_filter(imagem,'gauss')
        bdValue, new = flatten.flat_image(frame_denoise)
        crop = cropping.croppy_mona(new,bdValue)
        plt.figure()
        plt.subplot(121)
        plt.imshow(new, 'gray')
        plt.subplot(122)
        plt.imshow(crop, 'gray')
        #plt.plot(line, '.', lw=1)
        #plt.plot(line2, 'ro', lw=1)
        #plt.plot(poly, 'ro', lw=1)
        plt.show()


'''MAIN'''
# geraLBPFeatures('AMD1')
# geraLBPFeatures('NORMAL11')
# geraLBPFeatures('DME4')

gerarTrain(['./lbp_features/AMD1','./lbp_features/NORMAL11','./lbp_features/DME4'],['AMD','NORMAL','DME'],'AMDNORMALDME','AMD,NORMAL,DME','./lbp_features/')

# features_extraction.apply_hog(crop)
# features_extraction.hog2(crop)
# plt.figure()
# plt.subplot(121)
# plt.imshow(new, 'gray')
# plt.subplot(122)
# plt.imshow(crop, 'gray')
# plt.plot(line, '.', lw=1)
# #plt.plot(line2, 'ro', lw=1)
# #plt.plot(poly, 'ro', lw=1)
# plt.show()
#
