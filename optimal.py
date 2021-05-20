import cv2
import numpy as np
import  matplotlib.pyplot as plt
np.seterr(over='ignore')
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
img = cv2.imread("Threshold images\Henry_Moore_Sculpture_0252.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
op = optimal(gray)
cv2.imwrite("optimal.jpg",op)

plt.imshow(gray >= op)
plt.show()
