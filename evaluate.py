from image_NCut import *
import numpy as np
from scipy.sparse.linalg import ArpackError

arrayIoU = []
totExamples = 327
#totExamples = 10
top3 = [{1:0.0}, {1:0.0}, {1:0.0}]
bottom3 = [{1:1.0}, {1:1.0}, {1:1.0}]
skipped_images = []
for i in range(1, totExamples+1):
    img, mask = get_image(i)
    #apply gaussian blur to the image
    img = cv2.GaussianBlur(img, (5, 5), sigmaX=5, sigmaY=5)
    try:
        regions = imageNCut(img, mask, num_sections=500, num_cuts=500)
        prediction = defineHorse(regions, mask)
        arrayIoU.append(computeIoU(prediction, mask))
    except ArpackError:
        skipped_images.append(i)
        continue
    print(f'{i/totExamples*100:.2f}% done. \t Mean IoU: {np.mean(arrayIoU):.4f}', end='\r')
    for j in range(3):
        if arrayIoU[-1] > list(top3[j].values())[0]:
            top3[j] = {i:arrayIoU[-1]}
            break
        if arrayIoU[-1] < list(bottom3[j].values())[0]:
            bottom3[j] = {i:arrayIoU[-1]}
            break

print('\n')
print('Top 3 IoU:')
for i in top3:
    print(f'Image {list(i.keys())[0]} has IoU: {list(i.values())[0]}')
print('\nBottom 3 IoU:')
bottom3.reverse()
for i in bottom3:
    print(f'Image {list(i.keys())[0]} has IoU: {list(i.values())[0]}')
    
if len(skipped_images) > 0:
    print('\nSkipped images:')
for i in skipped_images:
    print(f'Image {i} was skipped due to ArpackError')
print('\nMean IoU: ', np.mean(arrayIoU))
print('Std dev: ', np.std(arrayIoU))

#show the histogram
import matplotlib.pyplot as plt

totSplit = 10
split = [0 for i in range(totSplit)]
categories = [f'{i*10}-{(i+1)*10}' for i in range(totSplit)]
for e in range(len(arrayIoU)):
    split[int(arrayIoU[e]*totSplit)] += 1

print(split)
plt.bar(categories, split)
ax = plt.gca()
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
plt.title('IoU distribution')
plt.xlabel('IoU')
plt.ylabel('Frequency')
#save the histogram
plt.savefig('IoU_distribution.png')
plt.show()