

def croppy(image):
    center_x = image.shape[0]/2
    center_y = image.shape[1]/2

    image_cropped = image[center_x-20:center_x+25,center_y - 75: center_y+75]
    print('form:',image_cropped.shape)
    return image_cropped

def croppy_gabor(image):
    center_x = image.shape[0]/2
    center_y = image.shape[1]/2

    image_cropped = image[center_x-50:center_x+51,center_y - 150: center_y+151]
    print('form:',image_cropped.shape)
    return image_cropped