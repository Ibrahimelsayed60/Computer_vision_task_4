# Computer_vision_task_4

## Otsu Threshold

Binarization plays an important role in digital image processing, mainly in computer vision applications. Thresholding is an efficient technique is crucial in binarization.

Otsu's thresholding method corresponds to the linear discriminant criteria that assumes that the image consists of only object (foreground) and background, and the heterogeneity and diversity of the background is ignored.  Otsu set the threshold so as to try to minimize the overlapping of the class distributions. according the definition of Otsu , the Otsu's method segments the image into two light and dark regions T0 and T1, where region T0 is a set of intensity level from 0 to t or in set notation T0 = {0, 1, ..., t} and region T1 = {t, t + 1, ..., l − 1, l} where t is the threshold value, l is the image maximum gray level (for instance 256).

The goal is to find the threshold value with the minimum entropy for sum of foreground and background. Otsu’s method determines the threshold value based on the statistical information of the image where for a chosen threshold value t the variance of clusters T0 and T1 can be computed.

Steps:

* Otsu Global Threshold:

  * calculate the threshold by minimizing the sum of the weighted group variances, where the weights are the probability of the respective groups,

  <img src="Threshold images\otsu_image_result\probability.png" alt="probability" style="zoom:80%;" />

  ​	Where r, c is index for row and column of the image, respectively, R and C is the number of rows and columns of the image, respectively. Wb(t), µb(t), and σ 2 b (t) as the weight, mean, and variance of class T0 with intensity value from 0 to t, respectively.

  ​	 σ 2 w as the weighed sum of group variances. The best threshold value t* is the value with the minimum within class variance. The within class variance defines as following:

<img src="Threshold images\otsu_image_result\weights.png" alt="weights" style="zoom:70%;" />

* Otsu Local threshold:

  * Then, we have to apply this threshold in global threshold as previous steps. But when we need to apply the threshold locally; we split the image to many size of window  and get the threshold for each window with Otsu algorithm. Then use each threshold of each window and apply in it and collect these thresholded windows in the original image.

  * Note:  We make the shape of image to be square to make the small windows applying in all pixels.

* Results:

  * when we apply this threshold in "MRIbrain1.jpg", we get:

    <img src="Threshold images\otsu_image_result\otsu_global_local_16_window.png" alt="otsu_global_local_16_window" style="zoom:70%;" />

  * when we apply Local threshold with different sizes of window as follow:

    * 16 window size:

      <img src="Threshold images\otsu_image_result\otsu_16_window.png" alt="otsu_16_window" style="zoom:70%;" />

    * 32 window size:

      <img src="Threshold images\otsu_image_result\otsu_32.png" alt="otsu_32" style="zoom:70%;" />

    * 64 window size:

      <img src="Threshold images\otsu_image_result\otsu_local_64.png" alt="otsu_local_64" style="zoom:70%;" />

    * 128 window size:

      <img src="Threshold images\otsu_image_result\otsu_local_128.png" alt="otsu_local_128" style="zoom:70%;" />

## Spectral Threshold:



# 2. Image Segmentation:

 Image segmentation is the classification of an image into different groups.

* Input Image :

## 2.1. k-means

* K-Means clustering algorithm is an unsupervised algorithm and it is used to segment the interest area from the background. It clusters, or partitions the given data into K-clusters or parts based on the K-centroids.

  <img src="Segmentation Images\k-means.png" style="zoom:40%;" />

## 2.2. Mean shift method

*  Mean shift treats the clustering problem by supposing that all points given represent samples from some underlying probability density function, with regions of high sample density corresponding to the local maxima of this distribution. To find these local maxima, the algorithm works by allowing the points to attract each other, via what might be considered a short-ranged “gravitational” force. Allowing the points to gravitate towards areas of higher density, one can show that they will eventually coalesce at a series of points, close to the local maxima of the distribution. Those data points that converge to the same local maxima are considered to be members of the same cluster.

  <img src="Segmentation Images\mean-shift.png" style="zoom:40%;" />


## 2.3. Agglemorative Method
The steps of the agglomerative clustering algorithm:
1. Define each data point as a different cluster, and the point itself as the centroid of the cluster. Also, assign
1 to item count, and initialize a label array showing centroid indexes in the centroid array.
2. Find the closest two centroids (minimum distance, maximum similarity).
3. Add the higher indexed cluster to the lower indexed cluster by combining centroids with average mean and summing item counts.
4. Replace all labels of the higher indexed cluster with labels of the lower indexed cluster in the labels array.
5. If the length of the centroids array is equal to K, end the process, and return label array and centroid array.
Else, go to 2nd step.

* Result:


## 2.4. Region Growing Method 
The basic idea of region growing is to assemble pixels with similar properties to form regions. Firstly, a seed pixel is found for each region to be segmented as the growth starting point, and then the seed pixel and the pixels in the surrounding neighborhood that have the same or similar properties as the seed pixel are merged into the region where the seed pixel is located. These new pixels are treated as new seeds to continue the above process until pixels that do not meet the conditions can be included. Such a region grows into.

* Steps:

1. Scan the image in sequence! Find the first pixel that does not belong, and set the pixel as (x0, Y0);

2. Taking (x0, Y0) as the center, consider the 4 neighborhood pixels (x, y) of (x0, Y0), if (x0, Y0) meets the growth criteria, merge (x, y) and (x0, Y0) in the same region, and push (x, y) onto the stack;

3. Take a pixel from the stack and return it to step 2 as (x0, Y0);

4. When the stack is empty! Return to step 1;

5. Repeat steps 1-4 until each point in the image has attribution. Growth ends.

* Result:







