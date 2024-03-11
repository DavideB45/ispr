import math
import cv2
from show_segmentation_sample import get_image

def super_pixelize(image, regions:int=100, render:bool=False):
    # Create the super pixels
    # compute region size based on number of clusters
    region_size = int(math.sqrt(image.shape[0]*image.shape[1]/regions))
    # apply SLIC and get the contours
    silc = cv2.ximgproc.createSuperpixelSLIC(image, region_size=region_size)
    silc.iterate()
    if render:
        print('Number of superpixels:', silc.getNumberOfSuperpixels())
        mask = silc.getLabelContourMask()
        mask = cv2.bitwise_not(mask)
        result = cv2.bitwise_and(image, image, mask=mask)
        cv2.imshow('Result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return silc.getLabels()

if __name__ == '__main__':
    img, _ = get_image(27)
    super_pixelize(img, regions=200, render=True)