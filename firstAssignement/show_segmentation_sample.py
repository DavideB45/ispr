import cv2
from typing import Tuple

# Function to show a sample of the segmentation dataset
def show_segmentation_sample(sampleID:int)->None:
    
    # Load the first image
    image = cv2.imread('weizmann_horse_db/horse/horse'+str(sampleID).zfill(3)+'.png')
    mask = cv2.imread('weizmann_horse_db/mask/horse'+str(sampleID).zfill(3)+'.png', cv2.IMREAD_GRAYSCALE)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] != 0:
                mask[i][j] = 0
            else:
                mask[i][j] = 255

    # Merge the image and the mask
    result = cv2.addWeighted(image, 0.9, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.9, 0.5)
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_image(imageID:int)->None:
    image = cv2.imread('weizmann_horse_db/horse/horse'+str(imageID).zfill(3)+'.png')
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_image(imageID:int)->Tuple[object, object]:
    image = cv2.imread('weizmann_horse_db/horse/horse'+str(imageID).zfill(3)+'.png')
    mask = cv2.imread('weizmann_horse_db/mask/horse'+str(imageID).zfill(3)+'.png', cv2.IMREAD_GRAYSCALE)
    return image, mask

if __name__ == '__main__':
    show_image(1)
    show_segmentation_sample(1)
    show_image(100)
    show_segmentation_sample(100)