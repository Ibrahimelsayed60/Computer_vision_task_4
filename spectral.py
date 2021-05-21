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


def spectral_local_threshold(image,block_size):
    if image.shape[0] < image.shape[1]:
        resized_image = cv2.resize(image,(image.shape[1],image.shape[1]))
    else:
        resized_image = cv2.resize(image,(image.shape[0],image.shape[0]))

    no_rows = resized_image.shape[0]
    no_cols = resized_image.shape[1]

    output_image = resized_image.copy()

    for r in range(0,resized_image.shape[0],block_size):
        for c in range(0,resized_image.shape[1],block_size):
            #### Blocks
            block = resized_image[r:min(r+block_size,no_rows),c:min(c+block_size,no_cols)]
            size_block = np.size(block)

            graylevel = range(0,256)
            ### Histogram 
            hist = [0] * 256
            for i in range(0,256):
                hist[i] = len(np.extract(np.asarray(block) == graylevel[i],block))

            variance = []
            s_max = (0,-np.inf)
            for bar1 in range(len(hist)):
                for bar2 in range(bar1, len(hist)): 
                    foreground_levels = np.extract(np.asarray(graylevel) >= bar2, graylevel)
                    background_levels = np.extract(np.asarray(graylevel) < bar1, graylevel)
                    midground_levels = np.extract((np.asarray(graylevel) > bar1) & (np.asarray(graylevel) < bar2) , graylevel)
                    foreground_hist = []
                    background_hist = []
                    midground_hist = []

                    back_weight = 0
                    mid_weight = 0
                    fore_weight = 0
                    ##### mean (m_g, m_f)
                    back_mean =   0
                    mid_mean = 0
                    fore_mean = 0
                    background_length = len(background_levels)
                    foreground_length = len(foreground_levels)
                    midground_length = len(midground_levels)

                    if background_length != 0:
                        for i in background_levels:
                            background_hist.append(hist[i])
                            total_back_hist = sum(background_hist)
                            back_weight = float(total_back_hist) / block_size

                        if back_weight != 0:

                            back_mean = np.sum(np.multiply(background_levels,np.asarray(background_hist))) / float(sum(background_hist))

                    if foreground_length != 0:
                        for i in foreground_levels:
                            foreground_hist.append(hist[i])
                            total_fore_hist = sum(foreground_hist)
                            fore_weight = float(total_fore_hist) / block_size

                        if fore_weight != 0:

                            fore_mean = np.sum(np.multiply(foreground_levels,np.asarray(foreground_hist))) / float(sum(foreground_hist))
                    if midground_length != 0:
                        for i in midground_levels:
                            midground_hist.append(hist[i])
                            total_mid_hist = sum(midground_hist)
                            mid_weight = float(total_mid_hist) / block_size

                        if fore_weight != 0:

                            fore_mean = np.sum(np.multiply(midground_levels,np.asarray(midground_hist))) / float(sum(midground_hist))

                    variance.append((back_weight * fore_weight  *  ((back_mean - fore_mean ) **2)) + ( mid_weight * fore_weight*((mid_mean-fore_mean)**2)) + (back_weight * mid_weight*((mid_mean-back_mean)**2)) ) 

                    if np.max(variance) > s_max[1]:
                        s_max = (bar1, np.max(variance))

                Threshold = (s_max[0]/255)*(image.max()-image.min()) + image.min()
                thresholded_block = global_threshold(block,Threshold)
                output_image[r:min(r+block_size,no_rows),c:min(c+block_size,no_cols)] = thresholded_block

    output_image = cv2.resize(output_image,(image.shape[0],image.shape[1]))
    return output_image
                    


def spectral_threshold(image):
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
    s_max = (0,-np.inf)
    for bar1 in range(len(hist)):
        for bar2 in range(bar1, len(hist)): 
            foreground_levels = np.extract(np.asarray(graylevel) >= bar2, graylevel)
            background_levels = np.extract(np.asarray(graylevel) < bar1, graylevel)
            midground_levels = np.extract((np.asarray(graylevel) > bar1) & (np.asarray(graylevel) < bar2) , graylevel)
            foreground_hist = []
            background_hist = []
            midground_hist = []

            back_weight = 0
            mid_weight = 0
            fore_weight = 0
            ##### mean (m_g, m_f)
            back_mean =   0
            mid_mean = 0
            fore_mean = 0
            background_length = len(background_levels)
            foreground_length = len(foreground_levels)
            midground_length = len(midground_levels)

            if background_length != 0:
                for i in background_levels:
                    background_hist.append(hist[i])
                    total_back_hist = sum(background_hist)
                    back_weight = float(total_back_hist) / imageSize

                if back_weight != 0:

                    back_mean = np.sum(np.multiply(background_levels,np.asarray(background_hist))) / float(sum(background_hist))

            if foreground_length != 0:
                for i in foreground_levels:
                    foreground_hist.append(hist[i])
                    total_fore_hist = sum(foreground_hist)
                    fore_weight = float(total_fore_hist) / imageSize

                if fore_weight != 0:

                    fore_mean = np.sum(np.multiply(foreground_levels,np.asarray(foreground_hist))) / float(sum(foreground_hist))
            if midground_length != 0:
                for i in midground_levels:
                    midground_hist.append(hist[i])
                    total_mid_hist = sum(midground_hist)
                    mid_weight = float(total_mid_hist) / imageSize

                if fore_weight != 0:

                    fore_mean = np.sum(np.multiply(midground_levels,np.asarray(midground_hist))) / float(sum(midground_hist))


            variance.append((back_weight * fore_weight  *  ((back_mean - fore_mean ) **2)) + ( mid_weight * fore_weight*((mid_mean-fore_mean)**2)) + (back_weight * mid_weight*((mid_mean-back_mean)**2)) ) 

            if np.max(variance) > s_max[1]:
                s_max = (bar1, np.max(variance))

    max_variance = np.max(variance)
    Threshold = (s_max[0]/255)*(image.max()-image.min()) + image.min() 
    return Threshold   

# image = cv2.imread("Threshold images\MRIbrain1.jpg",0)
# out = spectral_local_threshold(image,128)
# plt.imshow(out,cmap='gray')
# plt.show()