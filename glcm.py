

from skimage.feature import greycomatrix, greycoprops


def glcm_features(frame):
    features = []

    glcm = greycomatrix(frame, [5], [0], 256, symmetric=True, normed=True)
    features.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    features.append(greycoprops(glcm, 'correlation')[0, 0])
    features.append(greycoprops(glcm, 'homogeneity')[0, 0])
    features.append(greycoprops(glcm, 'ASM')[0, 0])
    features.append(greycoprops(glcm, 'energy')[0, 0])

    return features