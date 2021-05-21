import numpy as np
import matplotlib.pyplot as plt
from random import randint
import math

np.random.seed(42)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KMeans():

    def __init__(self, K=3, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # initialize 
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            
            # check if clusters have changed
            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # Classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)


    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters
    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # distances between each old and new centroids, fol all centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color='black', linewidth=2)

        plt.show()
    def cent(self):
        return self.centroids

class meanshift():
    def __init__(self, img):
        self.opImg = np.zeros(img.shape,np.uint8)
        self.H = 90
        self.Iter = 100

    def getNeighbors(self,seed,matrix):
        neighbors = []
        nAppend = neighbors.append
        sqrt = math.sqrt
        for i in range(0,len(matrix)):
            cPixel = matrix[i]
            d = sqrt(sum((cPixel-seed)**2))
            if(d<self.H):
                 nAppend(i)
        return neighbors

    def markPixels(self,neighbors,mean,matrix,cluster):
        for i in neighbors:
            cPixel = matrix[i]
            x=cPixel[3]
            y=cPixel[4]
            self.opImg[x][y] = np.array(mean[:3],np.uint8)
        return np.delete(matrix,neighbors,axis=0)

    def calculateMean(self,neighbors,matrix):
        neighbors = matrix[neighbors]
        r=neighbors[:,:1]
        g=neighbors[:,1:2]
        b=neighbors[:,2:3]
        x=neighbors[:,3:4]
        y=neighbors[:,4:5]
        mean = np.array([np.mean(r),np.mean(g),np.mean(b),np.mean(x),np.mean(y)])
        return mean

    def createFeatureMatrix(self,img):
        h,w,d = img.shape
        F = []
        FAppend = F.append
        for row in range(0,h):
            for col in range(0,w):
                r,g,b = img[row][col]
                FAppend([r,g,b,row,col])
        F = np.array(F)
        return F

    def performMeanShift(self,img):
        clusters = 0
        F = self.createFeatureMatrix(img)
        while(len(F) > 0):
            randomIndex = randint(0,len(F)-1)
            seed = F[randomIndex]
            initialMean = seed
            neighbors = self.getNeighbors(seed,F)

            if(len(neighbors) == 1):
                F=self.markPixels([randomIndex],initialMean,F,clusters)
                clusters+=1
                continue
            mean = self.calculateMean(neighbors,F)
            meanShift = abs(mean-initialMean)

            if(np.mean(meanShift)<self.Iter):
                F = self.markPixels(neighbors,mean,F,clusters)
                clusters+=1
        return self.opImg

class regionGrowing(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x
    def getY(self):
        return self.y

    def getGrayDiff(img,currentPoint,tmpPoint):
        return abs(int(img[currentPoint.x,currentPoint.y]) - int(img[tmpPoint.x,tmpPoint.y]))

    def selectConnects(p):
        if p != 0:
            connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
                Point(0, 1), Point(-1, 1), Point(-1, 0)]
        else:
            connects = [ Point(0, -1), Point(1, 0),Point(0, 1), Point(-1, 0)]
        return connects

    def regionGrow(img,seeds,thresh,p = 1):
        height, weight = img.shape
        seedMark = np.zeros(img.shape)
        seedList = []
        for seed in seeds:
            seedList.append(seed)
            label = 1
            connects = selectConnects(p)
        while(len(seedList)>0):
            currentPoint = seedList.pop(0)

            seedMark[currentPoint.x,currentPoint.y] = label
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
        if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
            grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))
        if grayDiff < thresh and seedMark[tmpX,tmpY] == 0:
             seedMark[tmpX,tmpY] = label
             seedList.append(Point(tmpX,tmpY))
        return seedMark




class AgglomerativeClustering(object):

    def __init__(self, cluster_count):
        self.cluster_count = cluster_count
        self.cluster_item_counts = np.array([])
        self.labels = np.array([])
        self.centroids = []

    def fit(self, data):
        if type(data) is not np.ndarray:
            data = np.array(data)

        self.cluster_item_counts = np.append(
            self.cluster_item_counts, np.ones(data.shape[0])
        )
        self.labels = np.append(
            self.labels, [len(self.centroids) + x for x in range(len(data))]
        )
        self.centroids += [d for d in data]

        while len(self.cluster_item_counts) != self.cluster_count:
            min_distance = np.inf
            i1, i2 = (0, 0)
            for i in range(len(self.cluster_item_counts)):
                distances = np.linalg.norm(self.centroids - self.centroids[i], axis=1)
                distances[distances == 0] = np.inf
                min_dist = np.min(distances)
                if min_dist < min_distance:
                    min_distance = min_dist
                    i1, i2 = i, np.argmin(distances)

            if i2 < i1:
                i1, i2 = i2, i1
            new_centroid = self.weigted_average(
                self.centroids[i1],
                self.centroids[i2],
                self.cluster_item_counts[i1],
                self.cluster_item_counts[i2],
            )
            self.centroids[i1] = new_centroid
            self.cluster_item_counts[i1] += self.cluster_item_counts[i2]
            self.cluster_item_counts = np.delete(self.cluster_item_counts, i2)
            del self.centroids[i2]
            self.labels[self.labels == i2] = i1
            self.labels[self.labels > i2] -= 1

        return np.array(self.centroids), self.labels

    def weigted_average(self, centroid1, centroid2, weight1, weight2):
        new_centroid = [0, 0, 0]
        for i in range(3):
            new_centroid[i] = np.average(
                [centroid1[i], centroid2[i]], weights=[weight1, weight2]
            )
        return np.array(new_centroid)


    def distortion(data, labels, centroids):
        """
        The distortion (clustering error) is the summation of distances between points and their own cluster centroids
        """
        n = data.shape[0]
        centroid_array = np.zeros((n, 3))
        for i in range(n):
            label = labels[i]
            cendroid = centroids[int(label)]
            centroid_array[i] = cendroid

        return np.linalg.norm(data - centroid_array)