import cv2
import numpy as np
import  matplotlib.pyplot as plt
np.seterr(over='ignore')

def global_threshold(image, thres_value):
    img = image> thres_value
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if img[i,j] == True :
                img[i,j] = 255
            else:
                img[i,j] = 0
    return img

def optimal(img):
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
def localThresh(img, blksize):
    if img.shape[0] != img.shape[1]:
        if img.shape[0] > img.shape[1]:
            resizedimg = cv2.resize(img,(img.shape[0],img.shape[0]))
        else:
            resizedimg = cv2.resize(img, (img.shape[1], img.shape[1]))
    else:
        resizedimg = img
    rows = resizedimg.shape[0]
    cols = resizedimg.shape[1]

    if blksize <= 2:
        print("Error in local thresholding , block size should be greater than 2 ! ")
        exit()

    if blksize > img.shape[0] and blksize > img.shape[1]:
        print("Error local thresholding , block size should be smaller than image size!")
        exit()

    outputImage = np.zeros(resizedimg.shape)

    for r in range (0, rows, blksize):
        for c in range(0,cols, blksize):
            block = resizedimg[r:min(r + blksize,rows), c:min(c + blksize, cols)]
            background = [block[0, 0], block[0, block.shape[1] - 1], block[block.shape[0] - 1, 0],
                          block[block.shape[0] - 1, block.shape[1] - 1]]
            background_mean = np.mean(background)
            foreground_mean = np.mean(block) - background_mean
            t = (background_mean + foreground_mean) / 2

            while True:
                oldthreshold = t
                newfore = block[np.where(block >= t)]
                newback = block[np.where(block < t)]
                if newback.size:
                    new_background_mean = np.mean(newback)
                else:
                    new_background_mean = 0
                if newfore.size:
                    new_foreground_mean = np.mean(newfore)
                else:
                    new_foreground_mean = 0
                # update threshold
                thresh = (new_background_mean + new_foreground_mean) / 2
                if oldthreshold == t:
                    break
            thresholdedBlock = np.zeros(block.shape)
            for row in range(0, block.shape[0]):
                for col in range(0, block.shape[1]):
                    if block[row, col] >= thresh:
                        thresholdedBlock[row, col] = 255
                    else:
                        thresholdedBlock[row, col] = 0
            outputImage[r:min(r + blksize, rows) , c:min(c + blksize, cols)] = thresholdedBlock
            outputImage = cv2.resize(outputImage, (img.shape[1], img.shape[0]))
            return outputImage


img = cv2.imread("Threshold images\Henry_Moore_Sculpture_0252.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
op = optimal(gray)
# cv2.imwrite("optimal.jpg",op)
plt.imshow(global_threshold(gray,200))
plt.show()
plt.imshow(gray >= op)
plt.show()




