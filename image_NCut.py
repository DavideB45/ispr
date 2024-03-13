import math
import cv2
import time
from matplotlib import pyplot as plt
import numpy as np
from skimage import graph, color
from sklearn.metrics import jaccard_score
from show_segmentation_sample import get_image

def imageNCut(image, mask, num_sections:int=1000, render:bool=False) -> cv2.typing.MatLike:
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
    if render: print('Time:', time.time()-start)
    out2 = color.label2rgb(lables2, image, kind='avg')

    if render:
        fig, ax = plt.subplots(nrows=2,ncols=2, sharex=True, sharey=True, figsize=(6, 8))
        ax[0][0].imshow(out1)
        ax[1][0].imshow(out2)
        ax[0][1].imshow(mask, cmap='gray')
        ax[1][1].imshow(image)
        plt.tight_layout()
        plt.show()
    #unique, counts = np.unique(lables2, return_counts=True)
    #counts, unique = zip(*sorted(zip(counts, unique), reverse=True))
    return lables2

def defineHorse(predictedClasses:cv2.typing.MatLike, groundTruth:cv2.typing.MatLike) -> cv2.typing.MatLike:
    begin = time.time()
    insideClasses = {}
    unique, counts = np.unique(predictedClasses, return_counts=True)
    validClasses = []
    for i in range(len(unique)):
        insideClasses[unique[i]] = [counts[i], 0]

    for i in range(len(predictedClasses)):
        for x in range(len(predictedClasses[0])):
            if groundTruth[i][x] == 1:
                j = predictedClasses[i][x]
                insideClasses[j][1] = insideClasses[j][1] + 1
    
    #decide what is mostly inside than outside (it will be a horse)
    for i in insideClasses:
        if(insideClasses[i][0] < insideClasses[i][1]*2):
            validClasses.append(i)
    #treat special case class 1 (the number used as mask)
    #same for class 0 (the bacgorund class)
    #alternative always sum 2 to everything
    max_ = max(unique)
    if 1 in unique:
        predictedClasses[predictedClasses == 1] = max_ + 1
        unique[unique == 1] = max_ + 1
        if 1 in validClasses:
            validClasses[validClasses == 1] = max_ + 1
    if 0 in unique:
        predictedClasses[predictedClasses == 0] = max_ + 2
        unique[unique == 0] = max_ + 2
        if 0 in validClasses:
            validClasses[validClasses == 0] = max_ + 2

    for i in unique:
        if i in validClasses:
            predictedClasses[predictedClasses == i] = 1
        else:
            predictedClasses[predictedClasses == i] = 0

    return predictedClasses

def computeIoU(predictedClasses:cv2.typing.MatLike, groundTruth:cv2.typing.MatLike, render:bool=False) -> float:
    begin = time.time()
    intersection = 0
    union = 0
    for i in range(len(predictedClasses)):
        for j in range(len(predictedClasses[0])):
            if groundTruth[i][j] == 1 or predictedClasses[i][j] == 1:
                union += 1
            if groundTruth[i][j] == 1 and predictedClasses[i][j] == 1:
                intersection += 1
    res = intersection/union
    end = time.time()
    if render:
        print('time: ', end - begin)
    return res

if __name__ == '__main__':
    imgEasy = [27, 128]
    for i in imgEasy:
        img, mask = get_image(i)
        regions = imageNCut(img, mask, num_sections=300, render=True)
        prediction = defineHorse(regions, mask)
        print('IoU: ',computeIoU(prediction, mask))