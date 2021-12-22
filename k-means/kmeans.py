
import numpy as np

#################################
# DO NOT IMPORT OHTER LIBRARIES
#################################

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data - numpy array of points
    :param generator: random number generator. Use it in the same way as np.random.
            In grading, to obtain deterministic results, we will be using our own random number generator.


    :return: a list of length n_clusters with each entry being the *index* of a sample
             chosen as centroid.
    '''
    p = generator.randint(0, n) #this is the index of the first center
    #############################################################################
    # TODO: implement the rest of Kmeans++ initialization. To sample an example
	# according to some distribution, first generate a random number between 0 and
	# 1 using generator.rand(), then find the the smallest index n so that the 
	# cumulative probability from example 1 to example n is larger than r.
    #############################################################################
    pts_cen = []
    pts_cen.append(x[p])
    centers = []
    centers.append(p)
    for k in range(n_cluster - 1):
        dist_centroid = []
        for i in range(len(x)):
            curr_min = float("inf")
            for j in range(len(pts_cen)):
                curr_cen = pts_cen[j]
                curr_dist = np.sum((x[i] - curr_cen)**2)
                if curr_dist < curr_min:
                    curr_min = curr_dist
            dist_centroid.append(curr_min)
        dist_sum = sum(dist_centroid)
        for m in range(len(dist_centroid)):
            dist_centroid[m] = dist_centroid[m] / dist_sum
        
        random_num = generator.rand()
        sum_all_prob = 0
        curr_idx = 0
        for v in range(len(dist_centroid)):
          sum_all_prob += dist_centroid[v]
          if sum_all_prob > random_num:
            curr_idx =  v
            break
        centers.append(curr_idx)
        pts_cen.append(x[curr_idx])

    # DO NOT CHANGE CODE BELOW THIS LINE
    return centers


# Vanilla initialization method for KMeans
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)



class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple in the following order:
                  - final centroids, a n_cluster X D numpy array, 
                  - a length (N,) numpy array where cell i is the ith sample's assigned cluster's index (start from 0), 
                  - number of times you update the curr_assign, an Int (at most self.max_iter)
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        self.generator.seed(42)
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        ###################################################################
        # TODO: Update means and membership until convergence 
        #   (i.e., average K-mean objective changes less than self.e)
        #   or until you have made self.max_iter updates.
        ###################################################################
        centroids = np.zeros([self.n_cluster, D])
        for i in range(self.n_cluster):
            centroids[i] = x[self.centers[i]]

        y = np.zeros(N)
        distor = np.sum([np.sum(np.power((x[y == i] - centroids[i]), 2)) for i in range(self.n_cluster)]) / N

        cnt = 0
        while cnt < self.max_iter:
            cnt += 1
            y = np.argmin(np.sum(((x - np.expand_dims(centroids, axis=1)) ** 2), axis=2), axis=0)
            new_distor = np.sum([np.sum(np.power((x[y == i] - centroids[i]), 2)) for i in range(self.n_cluster)]) / N
            if np.absolute(distor - new_distor) <= self.e:
                break
            distor = new_distor
            centroids_new = np.array([np.mean(x[y == i], axis=0) for i in range(self.n_cluster)])
            index = np.where(np.isnan(centroids_new))
            centroids_new[index] = centroids[index]
            centroids = centroids_new
        self.max_iter = cnt
        return centroids, y, self.max_iter

        


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Store following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (numpy array of length n_cluster)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        ################################################################
        # TODO:
        # - assign means to centroids (use KMeans class you implemented, 
        #      and "fit" with the given "centroid_func" function)
        # - assign labels to centroid_labels
        ################################################################
        curr_kmean = KMeans(self.n_cluster, self.max_iter, self.e, self.generator)
        centroids, curr_assign, curr_cnt = curr_kmean.fit(x, centroid_func)
        centroids = np.array(centroids)
        curr_y = [[] for i in range(self.n_cluster)]
        for i in range(N):
            curr_y[curr_assign[i]].append(y[i])

        centroid_labels = np.zeros([self.n_cluster])
        for i in range(self.n_cluster):
            centroid_labels[i] = np.argmax(np.bincount(curr_y[i]))

        
        # DO NOT CHANGE CODE BELOW THIS LINE
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        ##########################################################################
        # TODO:
        # - for each example in x, predict its label using 1-NN on the stored 
        #    dataset (self.centroids, self.centroid_labels)
        ##########################################################################
        norm_l2 = np.zeros([self.n_cluster, N])
        for c in range(self.n_cluster):
            norm_l2[c] = np.sqrt(np.sum(np.power((x - self.centroids[c]), 2), axis=1))
        curr_min_centroid = np.argmin(norm_l2, axis=0)
        labels = [[] for i in range(N)]
        for i in range(N):
            labels[i] = self.centroid_labels[curr_min_centroid[i]]

        return np.array(labels)
        




def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors (aka centroids)

        Return a new image by replacing each RGB value in image with the nearest code vector
          (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'
    ##############################################################################
    # TODO
    # - replace each pixel (a 3-dimensional point) by its nearest code vector
    ##############################################################################
    red, green, blue = image.shape
    img_reshape = image.reshape(red * green, blue)
    min_ner_idx = np.argmin(np.sum((np.power(
        img_reshape - np.expand_dims(code_vectors, axis=1), 2)), axis=2), axis=0)
    new_img = code_vectors[min_ner_idx].reshape(red, green, blue)
    return new_img
    
