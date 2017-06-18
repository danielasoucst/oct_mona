

def croppy(image,boundaryValue):
    center_x = image.shape[0]/2
    center_y = image.shape[1]/2

    image_cropped = image[boundaryValue-40:boundaryValue+5,center_y - 75: center_y+75]
    print('form:',image_cropped.shape)
    return image_cropped

def croppy_mona(image,boundaryValue):

    center_y = image.shape[1]/2

    image_cropped = image[boundaryValue-95:boundaryValue+6,center_y - 250: center_y+250]
    print('form:',image_cropped.shape)
    return image_cropped