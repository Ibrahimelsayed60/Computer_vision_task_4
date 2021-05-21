import cv2
import numpy as np
import  matplotlib.pyplot as plt
np.seterr(over='ignore')



def Optimal(img):
    background_sum = (img[0,0] + img[0,-1] + img[-1,0] +img[-1,-1])
    foreground_sum = np.sum(img) - background_sum
    background_mean = background_sum/4
    foreground_mean = foreground_sum / (np.size(img)-4)
    t = (foreground_mean+background_mean)/2
    while True:
        background_mean = np.mean(img[img<t])
        foreground_mean = np.mean(img[img>t])

        if (t == (background_mean+foreground_mean)/2):
            break
        t = (background_mean+foreground_mean)/2
    return  t

def globalThreshold(img):
    x = Optimal(img)
    img = img > x
    for i in range(0, img.shape[0], 1):
        for j in range(0, (img.shape[1]), 1):
            if img[i,j] == True :
                img[i,j] = 255
            else:
                img[i,j] = 0
    return img

def localOptimal(image, block_size):
    if image.shape[0] != image.shape[1]:
        if image.shape[0] > image.shape[1]:
            resizedImage = cv2.resize(image, (image.shape[0], image.shape[0]))
        else:
            resizedImage = cv2.resize(image, (image.shape[1], image.shape[1]))
    else:
        resizedImage = image
    rows = resizedImage.shape[0]
    cols = resizedImage.shape[1]

    if block_size <= 2:
        print("Error in local thresholding , block size should be greater than 2 ! ")
        exit()

    if block_size > image.shape[0] and block_size > image.shape[1]:
        print("Error local thresholding , block size should be smaller than image size!")
        exit()

    outputImage = np.zeros(resizedImage.shape)
    #------------------------------------ optimal thresholding algorithm------------------------------------------#

    for r in range(0, rows, block_size):
        for c in range(0, cols, block_size):
            # Extarct blocks
            block = resizedImage[r:min(r + block_size,rows), c:min(c + block_size, cols)]
            # get  initial background mean  (4corners)
            background = [block[0, 0], block[0, block.shape[1]-1], block[block.shape[0]-1, 0], block[block.shape[0]-1, block.shape[1]-1]]
            background_mean = np.mean(background)
            # get  initial foreground mean
            foreground_mean = np.mean(block) - background_mean
            # get  initial threshold
            thresh = (background_mean + foreground_mean) / 2.0
            while True:
                old_thresh = thresh
                new_foreground = block[np.where(block >= thresh)]
                new_background = block[np.where(block < thresh)]
                if new_background.size:
                    new_background_mean = np.mean(new_background)
                else:
                    new_background_mean = 0
                if new_foreground.size:
                    new_foreground_mean = np.mean(new_foreground)
                else:
                    new_foreground_mean = 0
                # update threshold
                thresh = (new_background_mean + new_foreground_mean) / 2
                if old_thresh == thresh:
                    break

            # convert to binary [ (0 , 255) only]
            thresholdedBlock = np.zeros(block.shape)
            for row in range(0, block.shape[0]):
                for col in range(0, block.shape[1]):
                    if block[row, col] >= thresh:
                        thresholdedBlock[row, col] = 255
                    else:
                        thresholdedBlock[row, col] = 0

            # fill the output image for each block
            outputImage[r:min(r + block_size, rows) , c:min(c + block_size, cols)] = thresholdedBlock

    # resize output  image back to original size
    outputImage = cv2.resize(outputImage, (image.shape[1], image.shape[0]))
    return outputImage

def togray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def showop(img):
    gs = togray(img)
    op = Optimal(gs)
    img = (gs >= op)
    return img

def showglob(img):
    gs = togray(img)
    op = globalThreshold(gs)
    return op

def showloc(img):
    gs = togray(img)
    op = localOptimal(gs,100)
    return op







