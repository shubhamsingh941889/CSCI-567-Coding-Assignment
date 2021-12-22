import numpy as np
from collections import Counter

############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################

class KNN:
    my_feature = None
    my_label = None
    def __init__(self, k, distance_function):
        """
        :param k: int
        :param distance_function
        """
        self.k = k
        self.distance_function = distance_function

    # TODO: save features and lable to self
    def train(self, features, labels):
        """
        In this function, features is simply training data which is a 2D list with float values.
        For example, if the data looks like the following: Student 1 with features age 25, grade 3.8 and labeled as 0,
        Student 2 with features age 22, grade 3.0 and labeled as 1, then the feature data would be
        [ [25.0, 3.8], [22.0,3.0] ] and the corresponding label would be [0,1]

        For KNN, the training process is just loading of training data. Thus, all you need to do in this function
        is create some local variable in KNN class to store this data so you can use the data in later process.
        :param features: List[List[float]]
        :param labels: List[int]
        """
        self.my_feature = list(features)
        self.my_label = list(labels)
        #raise NotImplementedError

    # TODO: find KNN of one i
    def get_k_neighbors(self, pt):
        """
        This function takes one single data i and finds the k nearest neighbours in the training set.
        It needs to return a list of labels of these k neighbours. When there is a tie in k_dist, 
		prioritize examples with a smaller index.
        :param i: List[float]
        :return:  List[int]
        """

        k_near = []
        for i in range(len(self.my_feature)):
            k_dist = self.distance_function(pt, self.my_feature[i])
            k_near.append((k_dist, self.my_label[i]))
        k_near.sort(key=lambda y: y[0])
        k_near_lab = []
        for i in range(self.k):
            k_near_lab.append(k_near[i][1])
        return k_near_lab
        #raise NotImplementedError
		
	# TODO: predict labels of a list of points
    def predict(self, features):
        """
        This function takes 2D list of test data points, similar to those from train function. Here, you need to process
        every test data i, reuse the get_k_neighbours function to find the nearest k neighbours for each test
        data i, find the majority of labels for these neighbours as the predicted label for that testing data i (you can assume that k is always a odd number).
        Thus, you will get N predicted label for N test data i.
        This function needs to return a list of predicted labels for all test data points.
        :param features: List[List[float]]
        :return: List[int]
        """
        p_lab = []
        for i in features:
            k_nl = self.get_k_neighbors(i)
            j = Counter(k_nl)
            most_common, count = j.most_common(1)[0]
            p_lab.append(most_common)
        return p_lab
        #raise NotImplementedError	


if __name__ == '__main__':
    print(np.__version__)
