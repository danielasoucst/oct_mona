# coding: utf-8
from matplotlib import pyplot as plt
import input,flatten,cropping,features_extraction,arffGenerator
import pickle

def gerarHOGFeatures(sujeito):
    lstImagens = input.load_image('./Publication_Dataset/'+sujeito+'/TIFFs/8bitTIFFs/')

    lstHOGFeatures=[]

    for imagem in lstImagens:
        frame_denoise = input.apply_filter(imagem,'anisotropic')
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
        frame_denoise = input.apply_filter(imagem,'anisotropic')
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
        frame_denoise = input.apply_filter(imagem,'anisotropic')
        bdValue, new = flatten.flat_image(frame_denoise)
        crop = cropping.croppy_mona(new,bdValue)
        crop = crop.astype(int)
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
        frame_denoise = input.apply_filter(imagem,'anisotropic')
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
    lstImagens = input.load_image('./Publication_Dataset/AMD6/TIFFs/8bitTIFFs/')


    for imagem in lstImagens:
        # frame_gauss = input.apply_filter(imagem,'gauss')
        frame_dif = input.apply_filter(imagem,'anisotropic')
        bdValue, new = flatten.flat_image(frame_dif)
        crop = cropping.croppy_mona(new,bdValue)

        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(frame_dif, 'gray')
        # plt.subplot(122)
        # plt.imshow(new, 'gray')
        # plt.plot(line, '.', lw=1)
        # #plt.plot(line2, 'ro', lw=1)
        # #plt.plot(poly, 'ro', lw=1)
        # plt.show()
def getClass(pasta):
    inicio = pasta[:3]
    if(inicio=='AMD'):
        return inicio
    if(inicio=='DME'):
        return  inicio
    if(inicio=='NOR'):
        return 'NORMAL'
    return None
def geraBaseFeatures():
    volumes = []
    volumes += ['AMD1','AMD2','AMD3','AMD4','AMD5','AMD6','AMD7','AMD8','AMD9','AMD10','AMD11','AMD12','AMD13','AMD14','AMD15']
    volumes += ['DME1','DME2','DME3','DME4','DME5','DME6','DME7','DME8','DME9','DME10','DME11','DME12','DME13','DME14','DME15']
    volumes += ['NORMAL1','NORMAL2','NORMAL3','NORMAL4','NORMAL5','NORMAL6','NORMAL7','NORMAL8','NORMAL9','NORMAL10','NORMAL11','NORMAL12','NORMAL13','NORMAL14','NORMAL15']

    for vol in volumes:
        print('Extraindo gabor features for: ',vol)
        geraGABORFeatures(vol)
    for vol in volumes:
        print('Extraindo glcm features for: ',vol)
        geraGLCMFeatures(vol)
    for vol in volumes:
        print('Extraindo lbp features for: ',vol)
        geraLBPFeatures(vol)
    print ("Fim...")

def loadFeatures():
    volumes = []
    volumes += ['AMD1', 'AMD2', 'AMD3', 'AMD4', 'AMD5', 'AMD6', 'AMD7', 'AMD8', 'AMD9', 'AMD10', 'AMD11', 'AMD12',
                'AMD13', 'AMD14', 'AMD15']
    volumes += ['DME1', 'DME2', 'DME3', 'DME4', 'DME5', 'DME6', 'DME7', 'DME8', 'DME9', 'DME10', 'DME11', 'DME12',
                'DME13', 'DME14', 'DME15']
    volumes += ['NORMAL1', 'NORMAL2', 'NORMAL3', 'NORMAL4', 'NORMAL5', 'NORMAL6', 'NORMAL7', 'NORMAL8', 'NORMAL9',
                'NORMAL10', 'NORMAL11', 'NORMAL12', 'NORMAL13', 'NORMAL14', 'NORMAL15']

    featuresGabor = []
    vetLabelsGabor = []
    featuresGLCM = []
    vetLabelsGLCM = []
    featuresLBP = []
    vetLabelsLBP = []

    '''GABOR + GLCM + LBP'''
    GGLfeatures = []
    vectLabelsGGL = []
    for vol in volumes:
        print('Carregando gabor+glcm features for: ', vol)
        fileObject = open('./gabor_features/' + vol, 'rb')
        features1 = pickle.load(fileObject)
        fileObject = open('./glcm_features/' + vol, 'rb')
        features2 = pickle.load(fileObject)
        fileObject = open('./lbp_features/' + vol, 'rb')
        features3 = pickle.load(fileObject)

        for i in range(0, len(features1)):
            features4 = features1[i]
            features4 += features2[i]
            features4 += features3[i]
            GGLfeatures.append(features4)

        vectLabelsGGL += [getClass(vol) for i in range(0, len(features1))]

    print ('Gerando arff file for gabor+ glcm + lbp')
    arffGenerator.createArffFile('./gabor_glcm_lbp_features/GABORGLCMLBPDATASET', GGLfeatures, vectLabelsGGL,
                                 'AMD,DME,NORMAL', len(GGLfeatures[0]))

    # vetLabelsGaborGLCM = []
    # GABORGLCMFeatures = []
    #
    # '''GABOR + GLCM'''
    # for vol in volumes:
    #     print('Carregando gabor+glcm features for: ', vol)
    #     fileObject = open('./gabor_features/' + vol, 'rb')
    #     features1 = pickle.load(fileObject)
    #     fileObject = open('./glcm_features/' + vol, 'rb')
    #     features2 = pickle.load(fileObject)
    #
    #     for i in range(0,len(features1)):
    #         features3 = features1[i]
    #         features3 += features2[i]
    #         GABORGLCMFeatures.append(features3)
    #
    #     vetLabelsGaborGLCM += [getClass(vol) for i in range(0,len(features1))]
    #
    # print ('Gerando arff file for gabor+ glcm')
    # arffGenerator.createArffFile('./gabor_glcm_features/GABORGLCMDATASET', GABORGLCMFeatures, vetLabelsGaborGLCM, 'AMD,DME,NORMAL',len(GABORGLCMFeatures[0]))
    #
    # vetLabelsGaborLBP = []
    # GABORLBPFeatures = []
    #
    # '''GABOR + LBP'''
    # for vol in volumes:
    #     print('Carregando gabor+lbp features for: ', vol)
    #     fileObject = open('./gabor_features/' + vol, 'rb')
    #     features1 = pickle.load(fileObject)
    #     fileObject = open('./lbp_features/' + vol, 'rb')
    #     features2 = pickle.load(fileObject)
    #
    #     for i in range(0, len(features1)):
    #         features3 = features1[i]
    #         features3 += features2[i]
    #         GABORLBPFeatures.append(features3)
    #
    #     vetLabelsGaborLBP += [getClass(vol) for i in range(0, len(features1))]
    #
    # print ('Gerando arff file for gabor+ lbp')
    # arffGenerator.createArffFile('./gabor_lbp_features/GABORLBPDATASET', GABORLBPFeatures, vetLabelsGaborLBP,
    #                              'AMD,DME,NORMAL', len(GABORLBPFeatures[0]))
    #
    # '''GLCM + LBP'''
    # vetLabelsGLCMLBP = []
    # GLCMLBPFeatures = []
    #
    # for vol in volumes:
    #     print('Carregando glcm+lbp features for: ', vol)
    #     fileObject = open('./glcm_features/' + vol, 'rb')
    #     features1 = pickle.load(fileObject)
    #     fileObject = open('./lbp_features/' + vol, 'rb')
    #     features2 = pickle.load(fileObject)
    #
    #     for i in range(0, len(features1)):
    #         features3 = features1[i]
    #         features3 += features2[i]
    #         GLCMLBPFeatures.append(features3)
    #
    #     vetLabelsGLCMLBP += [getClass(vol) for i in range(0, len(features1))]
    #
    # print ('Gerando arff file for glcm+ lbp')
    # arffGenerator.createArffFile('./glcm_lbp_features/GLCMLBPDATASET', GLCMLBPFeatures, vetLabelsGLCMLBP,
    #                              'AMD,DME,NORMAL', len(GLCMLBPFeatures[0]))

    # for vol in volumes:
    #     print('Carregando gabor features for: ', vol)
    #     fileObject = open('./gabor_features/' + vol, 'rb')
    #     featuresVolume = pickle.load(fileObject)
    #     featuresGabor += featuresVolume
    #     vetLabelsGabor += [getClass(vol) for i in range(0,len(featuresVolume))]
    #
    # print ('Gerando arff file for gabor')
    # arffGenerator.createArffFile('./gabor_features/GABORDATASET',featuresGabor,vetLabelsGabor,'AMD,DME,NORMAL',len(featuresGabor[0]))
    #
    # for vol in volumes:
    #     print('Extraindo glcm features for: ', vol)
    #     fileObject = open('./glcm_features/' + vol, 'rb')
    #     featuresVolume = pickle.load(fileObject)
    #     featuresGLCM += featuresVolume
    #     vetLabelsGLCM += [getClass(vol) for i in range(0,len(featuresVolume))]
    #
    # print ('Gerando arff file for glcm')
    # arffGenerator.createArffFile('./glcm_features/GLCMDATASET',featuresGLCM,vetLabelsGLCM,'AMD,DME,NORMAL',len(featuresGLCM[0]))
    #
    # for vol in volumes:
    #     print('Extraindo lbp features for: ', vol)
    #     fileObject = open('./lbp_features/' + vol, 'rb')
    #     featuresVolume = pickle.load(fileObject)
    #     featuresLBP += featuresVolume
    #     vetLabelsLBP += [getClass(vol) for i in range(0,len(featuresVolume))]
    #
    # print ('Gerando arff file for lbp')
    # arffGenerator.createArffFile('./lbp_features/LBPDATASET',featuresLBP,vetLabelsLBP,'AMD,DME,NORMAL',len(featuresLBP[0]))
    #

    print ("Fim...")
'''MAIN'''
# geraLBPFeatures('AMD1')
# geraLBPFeatures('NORMAL11')
# geraLBPFeatures('DME4')

# gerarTrain(['./lbp_features/AMD1','./lbp_features/NORMAL11','./lbp_features/DME4'],['AMD','NORMAL','DME'],'AMDNORMALDME','AMD,NORMAL,DME','./lbp_features/')
#geraBaseFeatures()
loadFeatures()

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
