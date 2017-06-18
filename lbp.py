import numpy as np
from skimage import feature

# settings for LBP
raio = 3
n_points = 8 * raio
eps=1e-7






def convertToLBP(window): #retorna o histograma da imagem ja converida para LBP
    lbp = feature.local_binary_pattern(window, n_points, raio, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3),
                             range=(0, n_points + 2))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist