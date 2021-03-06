import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """

    assert len(real_labels) == len(predicted_labels)
    orig_lbl = 0
    pred_lbl = 0
    aggregates = 0
    for orig_val, pred_val in zip(real_labels, predicted_labels):
        aggregates = aggregates + orig_val * pred_val
        orig_lbl = orig_lbl + orig_val
        pred_lbl = pred_lbl + pred_val
    return 2 * (float(aggregates) / float(orig_lbl + pred_lbl))
    raise NotImplementedError


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        sum = 0
        for i in range(len(point1)):
            sum = sum + (abs((point1[i] - point2[i])) ** 3)
        return sum ** (1./3)
        raise NotImplementedError

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        sum = 0
        for i in range(len(point1)):
            sum = sum + ((point1[i] -point2[i])**2)
        return sum**0.5
        raise NotImplementedError

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        sum_x =0
        sum_y = 0
        cros_sum =0
        for i in range(len(point1)):
            sum_x = sum_x+ point1[i]**2
            sum_y = sum_y + point2[i]**2
            cros_sum = cros_sum + point1[i]*point2[i]

        if sum_x**0.5 ==0 or sum_y**0.5 ==0:
            return 1
        else:
            return 1 - cros_sum/ ((sum_x**0.5) * (sum_y**0.5))

        raise NotImplementedError



class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best my_model with the highest f1-score on the given validation set.
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN my_model
        :param y_train: List[int] training labels to train your KNN my_model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and my_model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the my_model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
		(this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        curr_f1 = 0
        for i in distance_funcs:
            for k in range(1, min(31, len(x_train) + 1), 2):
                my_model = KNN(k, distance_funcs[i])
                my_model.train(x_train, y_train)
                pred_y = my_model.predict(x_val)
                new_f1 = f1_score(y_val, pred_y)
                if new_f1 > curr_f1:
                    self.best_k = k
                    self.best_distance_function = i
                    self.best_model = my_model
                    curr_f1 = new_f1
        
        #raise NotImplementedError

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN my_model, apply the scalers in scaling_classes to both of them. 
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN my_model
        :param y_train: List[int] train labels to train your KNN my_model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and my_model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.
        
        NOTE: When there is a tie, choose the my_model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        curr_f1 = 0
        for j in scaling_classes:
            norm = scaling_classes[j]()
            s_x_train = norm.__call__(x_train)
            s_x_val = norm.__call__(x_val)
            for i in distance_funcs:
                for k in range(1, min(31, len(x_train) + 1), 2):
                    my_model = KNN(k, distance_funcs[i])
                    my_model.train(s_x_train, y_train)
                    pred_y = my_model.predict(s_x_val)
                    new_f1 = f1_score(y_val, pred_y)
                    if new_f1 > curr_f1:
                        self.best_k = k
                        self.best_distance_function = i
                        self.best_scaler = j
                        self.best_model = my_model
                        curr_f1 = new_f1
        #raise NotImplementedError


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        ans = []
        #features = [[3, 4], [1, -1], [0, 0]]
        for i in features:
            sum = 0
            for j in range(len(i)):
                sum = sum + (i[j]**2)
            sum = sum**0.5
            if sum ==0:
                newList = i
            else:
                newList = [x / sum for x in i]
            ans.append(newList)
        print(ans)
        return ans

        raise NotImplementedError




class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        x = np.array(features)
        y_min = x.min(axis=0)
        y_max = x.max(axis=0)
        print(y_min[0],y_max)
        diff =[]
        for i in range(len(y_min)):
            dif_max_min =  y_max[i] - y_min[i]
            diff.append([y_min[i],dif_max_min])
        print(diff)

        for i in features:
            for j in range(len(i)):
                if diff[j][1]!=0:
                    print('g',i[j],1./3)
                    i[j] = float(i[j] - diff[j][0])/diff[j][1]
                else:
                    i[j] = 0

        return features





        raise NotImplementedError


