# coding: utf-8
from matplotlib import pyplot as plt
import input,flatten,cropping,features_extraction,arffGenerator
import pickle
import numpy as np
from sklearn.decomposition import PCA

def doPCA(features,num_comp):
    pca = PCA(n_components=num_comp)
    pca.fit(features)
    features_redu = pca.transform(features)
    print len(features_redu)
    return features_redu


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


def gera_GABOR_GLCM_LPB_features(sujeito):
    lstImagens = input.load_image('./Publication_Dataset/' + sujeito + '/TIFFs/8bitTIFFs/')

    volumeFeatures = []


    for imagem in lstImagens:
        frame_denoise = input.apply_filter(imagem, 'anisotropic')
        bdValue, new = flatten.flat_image(frame_denoise)
        crop = cropping.croppy_mona(new, bdValue)
        image_features = []

        image_features += features_extraction.apply_gabor(crop).tolist()
        crop2 = crop.astype(int)
        image_features += features_extraction.apply_glcm(crop2)
        image_features += features_extraction.apply_lbp(crop).tolist()

        volumeFeatures += image_features

        # features_extraction.apply_sift(crop)
        # lstSIFTFeatures.append(features_extraction.apply_sift(crop))

    print('gabor', len(volumeFeatures))
    # dictionary = features_extraction.apply_BOW(lstSIFTFeatures)
    fileObject = open('./gabor_glcm_lbp_repLine/' + sujeito, 'wb')
    if (fileObject != None):
        print('salvando...')
        pickle.dump(volumeFeatures, fileObject)
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

def getindice(pasta):
    inicio = pasta[:3]
    if(inicio=='AMD'):
        return pasta[3:]
    if(inicio=='DME'):
        return pasta[3:]
    if(inicio=='NOR'):
        return pasta[6:]
    return None

def geraBaseAllFeatures():
    volumes = []
    volumes += ['AMD1','AMD2','AMD3','AMD4','AMD5','AMD6','AMD7','AMD8','AMD9','AMD10','AMD11','AMD12','AMD13','AMD14','AMD15']
    volumes += ['DME1','DME2','DME3','DME4','DME5','DME6','DME7','DME8','DME9','DME10','DME11','DME12','DME13','DME14','DME15']
    volumes += ['NORMAL1','NORMAL2','NORMAL3','NORMAL4','NORMAL5','NORMAL6','NORMAL7','NORMAL8','NORMAL9','NORMAL10','NORMAL11','NORMAL12','NORMAL13','NORMAL14','NORMAL15']

    # for vol in volumes:
    #     print('Extraindo gabor features for: ',vol)
    #     geraGABORFeatures(vol)
    # for vol in volumes:
    #     print('Extraindo glcm features for: ',vol)
    #     geraGLCMFeatures(vol)
    # for vol in volumes:
    #     print('Extraindo lbp features for: ',vol)
    #     geraLBPFeatures(vol)

    for vol in volumes:
        print('Extraindo gabor glcm lbp features for: ', vol)
        gera_GABOR_GLCM_LPB_features(vol)

    print ("Fim...")

def geraBaseFeaturesTest():
    volumes = []
    volumes += ['AMD1','AMD2','AMD3','AMD4','AMD5','AMD7','AMD8','AMD9','AMD10','AMD12','AMD13','AMD14']
    volumes += ['DME1','DME2','DME3','DME4','DME6','DME7','DME8','DME9','DME12','DME13','DME14','DME15']
    volumes += ['NORMAL1','NORMAL2','NORMAL3','NORMAL4','NORMAL5','NORMAL6','NORMAL7','NORMAL8','NORMAL9','NORMAL11','NORMAL13','NORMAL14']

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

def loadFeaturesTest():
    volumes =['AMD6','AMD11','AMD15','DME5','DME10','DME11','NORMAL10','NORMAL12','NORMAL15']

    for vol in volumes:
        GGLfeatures = []
        vectLabelsGGL = []
        print('Carregando gabor+glcm+lbp features for: ', vol)
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
        arffGenerator.createArffFile('./gabor_glcm_lbp_features/'+vol+'DATASET', GGLfeatures, vectLabelsGGL,
                                     'AMD,DME,NORMAL', len(GGLfeatures[0]))
def geraAllArffsTrain():
    volumes = []
    volumes += ['AMD1', 'AMD2', 'AMD3', 'AMD4', 'AMD5', 'AMD6', 'AMD7', 'AMD8', 'AMD9', 'AMD10', 'AMD11', 'AMD12',
                'AMD13', 'AMD14', 'AMD15']
    volumes += ['DME1', 'DME2', 'DME3', 'DME4', 'DME5', 'DME6', 'DME7', 'DME8', 'DME9', 'DME10', 'DME11', 'DME12',
                'DME13', 'DME14', 'DME15']
    volumes += ['NORMAL1', 'NORMAL2', 'NORMAL3', 'NORMAL4', 'NORMAL5', 'NORMAL6', 'NORMAL7', 'NORMAL8', 'NORMAL9',
                'NORMAL10', 'NORMAL11', 'NORMAL12', 'NORMAL13', 'NORMAL14', 'NORMAL15']
    indiceTeste = 1
    for i in range(0,15):
        '''GABOR + GLCM + LBP'''
        GGLfeatures = []
        vectLabelsGGL = []

        GGLfeaturesTest = []
        vectLabelsGGLTest = []

        for j in range(0,len(volumes)):
            vol = volumes[j]

            if(getindice(vol) != str(indiceTeste)):
                '''ARFF TREINO'''
                fileObject = open('./gabor_features/' +volumes[j], 'rb')
                features1 = pickle.load(fileObject)
                fileObject = open('./glcm_features/' + volumes[j], 'rb')
                features2 = pickle.load(fileObject)
                fileObject = open('./lbp_features/' + volumes[j], 'rb')
                features3 = pickle.load(fileObject)

                for k in range(0, len(features1)):
                    features4 = []
                    features4 += features1[k]
                    features4 += features2[k]
                    features4 += features3[k]
                    GGLfeatures.append(features4)


                vectLabelsGGL += [getClass(volumes[j]) for x in range(0, len(features1))]
            else:
                print('nao entrou',vol)
                '''ARFF TESTE'''

                fileObject = open('./gabor_features/' + vol, 'rb')
                features1 = pickle.load(fileObject)
                fileObject = open('./glcm_features/' + vol, 'rb')
                features2 = pickle.load(fileObject)
                fileObject = open('./lbp_features/' + vol, 'rb')
                features3 = pickle.load(fileObject)

                for k in range(0, len(features1)):
                    features4 = []
                    features4 += features1[k]
                    features4 += features2[k]
                    features4 += features3[k]
                    GGLfeaturesTest.append(features4)

                vectLabelsGGLTest += [getClass(vol) for x in range(0, len(features1))]




        print ('Gerando arff file for gabor+ glcm + lbp',len(GGLfeatures),len(vectLabelsGGL))
        arffGenerator.createArffFile('./train/GABORGLCMLBP'+str(indiceTeste)+'TRAINDATASET', GGLfeatures, vectLabelsGGL,
                                     'AMD,DME,NORMAL', len(GGLfeatures[0]))
        arffGenerator.createArffFile('./train/GABORGLCMLBP' + str(indiceTeste) + 'TESTDATASET', GGLfeaturesTest, vectLabelsGGLTest,
                                     'AMD,DME,NORMAL', len(GGLfeatures[0]))
        indiceTeste += 1

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

    '''GABOR + GLCM + LBP unica instancia'''
    GGLfeatures = []
    vectLabelsGGL = []
    lstTam = []
    for vol in volumes:
        print('Carregando gabor+glcm features for: ', vol)
        fileObject = open('./gabor_glcm_lbp_repLine/' + vol, 'rb')
        features1 = pickle.load(fileObject)
        GGLfeatures.append(np.asarray(features1).astype(np.float32))
        vectLabelsGGL.append(getClass(vol))
        lstTam.append(len(features1))
    print lstTam
    print np.min(lstTam),len(GGLfeatures[0])
    print GGLfeatures[0].dtype
    GGLfeatures = features_extraction.apply_BOW(GGLfeatures,np.min(lstTam))
    print ('Gerando arff file for gabor+ glcm + lbp',len(GGLfeatures),len(vectLabelsGGL))
    arffGenerator.createArffFile('./gabor_glcm_lbp_repLine/GABORGLCMLBPLineDATASET', GGLfeatures, vectLabelsGGL,
                                 'AMD,DME,NORMAL', len(GGLfeatures[0]))
    # '''GABOR + GLCM + LBP'''
    # GGLfeatures = []
    # vectLabelsGGL = []
    # for vol in volumes:
    #     print('Carregando gabor+glcm features for: ', vol)
    #     fileObject = open('./gabor_features/' + vol, 'rb')
    #     features1 = pickle.load(fileObject)
    #     fileObject = open('./glcm_features/' + vol, 'rb')
    #     features2 = pickle.load(fileObject)
    #     fileObject = open('./lbp_features/' + vol, 'rb')
    #     features3 = pickle.load(fileObject)
    #
    #     for i in range(0, len(features1)):
    #         features4 = []
    #         features4 += features1[i]
    #         features4 += features2[i]
    #         features4 += features3[i]
    #         GGLfeatures.append(features4)
    #
    #
    #     vectLabelsGGL += [getClass(vol) for i in range(0, len(features1))]
    #
    # GGLfeatures = doPCA(GGLfeatures, 100)
    # print ('Gerando arff file for gabor+ glcm + lbp')
    # arffGenerator.createArffFile('./gabor_glcm_lbp_features/GABORGLCMLBPPCADATASET', GGLfeatures, vectLabelsGGL,
    #                              'AMD,DME,NORMAL', len(GGLfeatures[0]))

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
geraAllArffsTrain()

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
