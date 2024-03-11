import math
import cv2
import time
from matplotlib import pyplot as plt
import numpy as np
from skimage import graph, color
from show_segmentation_sample import get_image

def imageNCut(image, mask, num_sections:int=1000, render:bool=False):
    # Create the super pixels
    # compute region size based on number of clusters
    region_size = int(math.sqrt(image.shape[0]*image.shape[1]/num_sections))
    # apply SLIC and get the contours
    slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=region_size)
    slic.iterate()
    labels1 = slic.getLabels()
    out1 = color.label2rgb(labels1, img, kind='avg', bg_label=0)
    gr = graph.rag_mean_color(image, labels1, mode='similarity')
    start = time.time()
    lables2 = graph.cut_normalized(labels=labels1, rag=gr, thresh=0.01, num_cuts=10, max_edge=0.8)
    print('Time:', time.time()-start)
    out2 = color.label2rgb(lables2, image, kind='avg')

    fig, ax = plt.subplots(nrows=2,ncols=2, sharex=True, sharey=True, figsize=(6, 8))

    ax[0][0].imshow(out1)
    ax[1][0].imshow(out2)
    ax[0][1].imshow(mask, cmap='gray')
    

    # get more frequent labels
    # used together with class position
    unique, counts = np.unique(lables2, return_counts=True)
    #sort the labels by frequency
    counts, unique = zip(*sorted(zip(counts, unique), reverse=True))

    best_classes = list(unique[:5])
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if lables2[i,j] in best_classes:
                if i < 10:
                    best_classes.remove(lables2[i,j])
                elif j > image.shape[1]-10:
                    best_classes.remove(lables2[i,j])
                elif i > image.shape[0]-10:
                    best_classes.remove(lables2[i,j])
                elif j < 10:
                    best_classes.remove(lables2[i,j])
                elif counts[unique.index(lables2[i,j])] < 2000:
                    best_classes.remove(lables2[i,j])
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if lables2[i,j] in best_classes:
                image[i,j] = [0,255,255]
    ax[1][1].imshow(image)
    #for a in ax:
    #    a.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    imgEasy = [27, 128]
    for i in imgEasy:
        img, mask = get_image(i)
        imageNCut(img, mask, num_sections=300, render=True)