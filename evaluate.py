from image_NCut import *
import numpy as np

arrayIoU = []
totExamples = 3
for i in [27,128]:
    img, mask = get_image(i)
    regions = imageNCut(img, mask, num_sections=300, render=True)
    prediction = defineHorse(regions, mask)
    arrayIoU.append(computeIoU(prediction, mask))

print('Mean IoU: ', np.mean(arrayIoU))
print('Std IoU: ', np.std(arrayIoU))
