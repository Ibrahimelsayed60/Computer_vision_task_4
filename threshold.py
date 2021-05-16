########################### Threshold Function ##################
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import copy

def global_threshold(image,threshold):
    binary = image > threshold
    for i in range(0,binary.shape[0],1):
        for j in range(0,(binary.shape[1]),1):
            if binary[i][j] == True:
                binary[i][j] = 256
            else:
                binary[i][j]=0
    return binary

def local_threshold(input_img):
    h, w = input_img.shape
    S = w/8
    s2 = S/2
    T = 15.0
        #integral img
    int_img = np.zeros_like(input_img, dtype=np.uint32)
    for col in range(w):
        for row in range(h):
            int_img[row,col] = input_img[0:row,0:col].sum()
        #output img
    out_img = np.zeros_like(input_img)    
    for col in range(w):
        for row in range(h):
                #SxS region
            y0 = int(max(row-s2, 0))
            y1 = int(min(row+s2, h-1))
            x0 = int(max(col-s2, 0))
            x1 = int(min(col+s2, w-1))
            count = (y1-y0)*(x1-x0)
            sum_ = int_img[y1, x1]-int_img[y0, x1]-int_img[y1, x0]+int_img[y0, x0]
            if input_img[row, col]*count < sum_*(100.-T)/100.:
                out_img[row,col] = 0
            else:
                out_img[row,col] = 255
    return out_img


def otsu(gray):
    pixel_number = gray.shape[0] * gray.shape[1]
    mean_weigth = 1.0/pixel_number
    his, bins = np.histogram(gray, np.array(range(0, 256)))
    final_thresh = -1
    final_value = -1
    for t in bins[1:-1]: # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
        Wb = np.sum(his[:t]) * mean_weigth
        Wf = np.sum(his[t:]) * mean_weigth

        mub = np.mean(his[:t])
        muf = np.mean(his[t:])

        value = Wb * Wf * (mub - muf) ** 2

        #print("Wb", Wb, "Wf", Wf)
        #print("t", t, "value", value)

        if value > final_value:
            final_thresh = t
            final_value = value
    final_img = gray.copy()
    #print(final_thresh)
    final_img[gray > final_thresh] = 255
    final_img[gray < final_thresh] = 0
    return final_img

def otsu_global_threshold(image):
    no_rows = image.shape[0]
    no_cols = image.shape[1]
    imageSize = no_rows * no_cols
    graylevel = range(0,256)
    ### Histogram 
    hist = [0] * 256
    for i in range(0,256):
        hist[i] = len(np.extract(np.asarray(image) == graylevel[i],image))
    #counts,histo = np.histogram(image)
    variance = []
    for i in range(256):
        threshold = i
        background_gray_level =  np.extract(np.asarray(graylevel) < threshold, graylevel)
        foreground_gray_level =  np.extract(np.asarray(graylevel) >= threshold, graylevel)
        background_hist = []
        foreground_hist = []

        ##### Weights(W_g, W_f)
        back_weight = 0
        fore_weight = 0
        ##### mean (m_g, m_f)
        back_mean =   0
        fore_mean =   0

        background_length = len(background_gray_level)
        foreground_length = len(foreground_gray_level)

        if background_length != 0:
            for i in background_gray_level:
                background_hist.append(hist[i])
                total_back_hist = sum(background_hist)
                back_weight = float(total_back_hist) / imageSize

            if back_weight != 0:

                back_mean = np.sum(np.multiply(background_gray_level,np.asarray(background_hist))) / float(sum(background_hist))


        if foreground_length != 0:
            for i in foreground_gray_level:
                foreground_hist.append(hist[i])
                total_fore_hist = sum(foreground_hist)
                fore_weight = float(total_fore_hist) / imageSize

            if fore_weight != 0:

                fore_mean = np.sum(np.multiply(foreground_gray_level,np.asarray(foreground_hist))) / float(sum(foreground_hist))

        variance.append(back_weight * fore_weight * ((back_mean - fore_mean) **2)) 

    max_variance = np.max(variance)
    Threshold= variance.index(max_variance)
    outputImage = image.copy()
    print(Threshold)
    outputImage = global_threshold(image,Threshold)
    return outputImage

def otsu_local_threshold(image,block_size):
    pass



def otsu_threshold(img):
    min_intensity = img.min()
    max_intensity = img.max()
    image = ((img - min_intensity) * (1 / (max_intensity - min_intensity))*255).astype('uint8')

    ##### Get the histogram
    pixel_counts = [np.sum(image==i) for i in range(256)]
    sn_max = (0,-np.inf)

    ###### For loop to apply the equations of probability
    for threshold in range(256):
        n1 = sum(pixel_counts[:threshold])
        n2 = sum(pixel_counts[threshold:])

        mean1 = sum([i * pixel_counts[i] for i in range(0,threshold)]) / n1 if n1 > 0 else 0
        mean2 = sum([i * pixel_counts[i] for i in range(threshold,256)]) / n2 if n2 > 0 else 0

        ###### Calulate the variance
        s = n1 * n2 * (mean1 - mean2) ** 2
        if s > sn_max[1]:
            sn_max = (threshold,s)

    t = (sn_max[0]/255)*(max_intensity-min_intensity) + min_intensity 
    return t   



# image = cv2.imread("Threshold images\Pyramids2.jpg",0)
# out = otsu_global_threshold(image)
# #thres = otsu_threshold(image)
# #print(thres)
# plt.imshow(out,cmap='gray')
# plt.show()


