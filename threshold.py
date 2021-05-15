########################### Threshold Function ##################
import cv2
import matplotlib.pyplot as plt
import numpy as np

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

def otsu_global(image,threshold):
    binary = image > threshold
    for i in range(0,binary.shape[0],1):
        for j in range(0,(binary.shape[1]),1):
            if binary[i][j] == True:
                binary[i][j] = 256
            else:
                binary[i][j]=0
    return binary


def otsu_local(input_img,threshold):
    h, w = input_img.shape
    S = w/8
    s2 = S/2
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
            if input_img[row, col]*count < sum_*(100.-threshold)/100.:
                out_img[row,col] = 0
            else:
                out_img[row,col] = 255
    return out_img


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





