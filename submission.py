import numpy as np
import pandas as pd

X = np.load('example_X.npy')

Y = np.load('example_y.npy')

treatment = np.load('example_treatment.npy')

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, ddp = None, value=None):
        ''' constructor ''' 
        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.ddp = ddp
        
        # for leaf node
        self.value = value

class UpliftTreeRegressor():
    def __init__(self, max_depth = 3, min_samples_leaf = 1000, min_samples_leaf_treated = 300, min_samples_leaf_control=300):
        
        # initializing the root of the tree
        self.root = None
        
        #stopping conditions
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_leaf_treated = min_samples_leaf_treated
        self.min_samples_leaf_control = min_samples_leaf_control
        
    def build_tree(self, X, treatment, Y, depth = 0):
        ''' function to recursively build the tree '''
        num_samples, num_features = np.shape(X)
        dataset = np.column_stack((X,treatment,Y))
        best_split = {}
        max_ddp = -float("inf")
        if (depth <= self.max_depth):
            for feature_index in range(num_features):
                column_values = data_array[:,feature_index]
                unique_values  = np.unique(column_values)
                if len(unique_values) >10: 
                    percentiles = np.percentile(column_values, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97]) 
                else: 
                    percentiles = np.percentile(unique_values, [10, 50, 90]) 
                threshold_options = np.unique(percentiles)
                for threshold in threshold_options:
                    # get current split
                    dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                    if (np.shape(dataset_left)[0] <  self.min_samples_leaf or np.shape(dataset_right)[0] < self.min_samples_leaf):
                        continue
                    #get splits for treatment and control groups 
                    dataset_left_t = dataset_left[np.where(dataset_left[:,-2]==1)]
                    dataset_left_c = dataset_left[np.where(dataset_left[:,-2]==0)]
                    #check that the stopping conditions are met
                    if (np.shape(dataset_left_t)[0] < self.min_samples_leaf_treated or np.shape(dataset_left_c)[0] < self.min_samples_leaf_control):
                        continue

                    dataset_right_t = dataset_right[np.where(dataset_right[:,-2]==1)]
                    dataset_right_c = dataset_right[np.where(dataset_right[:,-2]==0)]
                    if (np.shape(dataset_right_t)[0] < self.min_samples_leaf_treated or np.shape(dataset_right_c)[0] < self.min_samples_leaf_control):
                        continue
                    #compute delta delta probability
                    ddp = (np.mean(dataset_left_t[:,-1]) - np.mean(dataset_left_c[:,-1])) - (np.mean(dataset_right_t[:,-1]) - np.mean(dataset_right_c[:,-1]))
                    #update the best split
                    if ddp > max_ddp:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["ddp"] = ddp

                        left_subtree = self.build_tree(best_split["dataset_left"][:,:num_features],best_split["dataset_left"][:,num_features:-1],best_split["dataset_left"][:,-1], depth+1)
                        right_subtree = self.build_tree(best_split["dataset_right"][:,:num_features],best_split["dataset_right"][:,num_features:-1],best_split["dataset_right"][:,-1], depth+1)
                        
                        return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree, best_split["ddp"])
        leaf_value = np.mean(Y)
        return Node(value = leaf_value)

    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.ddp)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
        
    def fit(self, X, treatment, Y):
        self.root = self.build_tree(X, treatment, Y)
        
    def make_prediction(self, x, tree):
        ''' function to predict new dataset '''
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
        
    def predict(self, X):
        preditions = [self.make_prediction(x, self.root) for x in X]
        return predictions


regressor = UpliftTreeRegressor()
regressor.fit(X,treatment,Y)
regressor.print_tree()

