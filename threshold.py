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
    
    ## Checking the dimension of image comparing to block size
    # resize the input image
    if image.shape[0] < image.shape[1]:
        resized_image = cv2.resize(image,(image.shape[1],image.shape[1]))
    else:
        resized_image = cv2.resize(image,(image.shape[0],image.shape[0]))

    no_rows = resized_image.shape[0]
    no_cols = resized_image.shape[1]

    if block_size > resized_image.shape[0] and block_size > resized_image.shape[1]:
        print("You can not apply local thresold in image")
        return 0

    output_image = resized_image.copy()

    #### Then apply the otsu algorithm
    ### The difference between local and global is we divide the image into windows
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
                        back_weight = float(total_back_hist) / size_block

                    if back_weight != 0:

                        back_mean = np.sum(np.multiply(background_gray_level,np.asarray(background_hist))) / float(sum(background_hist))


                if foreground_length != 0:
                    for i in foreground_gray_level:
                        foreground_hist.append(hist[i])
                        total_fore_hist = sum(foreground_hist)
                        fore_weight = float(total_fore_hist) / size_block

                    if fore_weight != 0:

                        fore_mean = np.sum(np.multiply(foreground_gray_level,np.asarray(foreground_hist))) / float(sum(foreground_hist))

                variance.append(back_weight * fore_weight * ((back_mean - fore_mean) **2))
            
            max_variance = np.max(variance)
            Threshold= variance.index(max_variance)

            thresholded_block = global_threshold(block,Threshold)

            output_image[r:min(r+block_size,no_rows),c:min(c+block_size,no_cols)] = thresholded_block

    output_image = cv2.resize(output_image,(image.shape[0],image.shape[1]))
    return output_image       

    





# image = cv2.imread("Threshold images\DNA_011.TIF",0)
# out = otsu_local_threshold(image,256)
# #thres = otsu_threshold(image)
# #print(thres)
# plt.imshow(out,cmap='gray')
# plt.show()


