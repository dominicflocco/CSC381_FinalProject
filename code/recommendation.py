'''
CSC381: Building a simple Recommender System

The final code package is a collaborative programming effort between the
CSC381 student(s) named below, the class instructor (Carlos Seminario), and
source code from Programming Collective Intelligence, Segaran 2007.
This code is for academic use/purposes only.

CSC381 Programmer/Researcher: Dominic Flocco

'''

#from curses.ascii import SI
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from math import *
import math
import copy
import pickle
import pandas as pd
from numpy.linalg import solve ## needed for als
from sklearn.metrics import mean_squared_error
import timeit
from time import time
from copy import deepcopy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from keras.models import load_model
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate, Dropout, BatchNormalization
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sympy import evaluate
warnings.filterwarnings('ignore')
# from mf_sgd_als_class import ExplicitMF
from NeuMF import NeuMF
# TFIDF 
TFIDF_SIG_THRESHOLD = 0.0
# Item-based distance
IBD_SIG_THRESHOLD = 0.0
IBD_SIG_WEIGHT = 50 
# Item-based pearson 
IBP_SIG_THRESHOLD = 0.0
IBP_SIG_WEIGHT = 50
# User-based distance 
UBD_SIG_THRESHOLD = 0.0
UBD_SIG_WEIGHT = 25
# User-based pearson 
UBP_SIG_THRESHOLD = 0.3
UBP_SIG_WEIGHT = 1

HYBRID_WEIGHT = 1.0
ITERATIONS = 20
# MF ALS 
ALS_FACTORS = 2
ALS_REG = 0.1
# MF SGD 
SGD_FACTORS = 200
SGD_LEARNING_RATE = 0.02
SGD_REG = 0.02


class ExplicitMF():
    def __init__(self, 
                 ratings,
                 n_factors=40,
                 learning='sgd',
                 sgd_alpha = 0.1,
                 sgd_beta = 0.1,
                 sgd_random = False,
                 item_fact_reg=0.0, 
                 user_fact_reg=0.0,
                 item_bias_reg=0.0,
                 user_bias_reg=0.0,
                 max_iters = 20,
                 verbose=True):
        """
        Train a matrix factorization model to predict empty 
        entries in a matrix. The terminology assumes a 
        ratings matrix which is ~ user x item
        
        Params
        ======
        ratings : (ndarray)
            User x Item matrix with corresponding ratings
            Note: can be full ratings matrix or train matrix
        
        n_factors : (int)
            Number of latent factors to use in matrix factorization model
            
        learning : (str)
            Method of optimization. Options include 'sgd' or 'als'.
        
        sgd_alpha: (float)
            learning rate for sgd
            
        sgd_beta:  (float)
            regularization for sgd
            
        sgd_random: (boolean)
            False makes use of random.seed(0)
            False means don't make it random (ie, make it predictable)
            True means make it random (ie, changee everytime code is run)
        
        item_fact_reg : (float)
            Regularization term for item latent factors
            Note: currently, same value as user_fact_reg
        
        user_fact_reg : (float)
            Regularization term for user latent factors
            Note: currently, same value as item_fact_reg
            
        item_bias_reg : (float)
            Regularization term for item biases
            Note: for later use, not used currently
        
        user_bias_reg : (float)
            Regularization term for user biases
            Note: for later use, not used currently
            
        max_iters : (integer)
            maximum number of iterations
        
        verbose : (bool)
            Whether or not to printout training progress
            
            
        Original Source info: 
            https://blog.insightdatascience.com/explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea#introsgd
            https://gist.github.com/EthanRosenthal/a293bfe8bbe40d5d0995#file-explicitmf-py
        """
        
        self.ratings = ratings 
        self.n_users, self.n_items = ratings.shape
        self.n_factors = n_factors
        self.item_fact_reg = item_fact_reg
        self.user_fact_reg = user_fact_reg
        self.item_bias_reg = item_bias_reg 
        self.user_bias_reg = user_bias_reg 
        self.learning = learning
        if self.learning == 'als':
            np.random.seed(0)
        if self.learning == 'sgd':
            self.sample_row, self.sample_col = self.ratings.nonzero()
            self.n_samples = len(self.sample_row)
            self.sgd_alpha = sgd_alpha # sgd learning rate, alpha
            self.sgd_beta = sgd_beta # sgd regularization, beta
            self.sgd_random = sgd_random # randomize
            if self.sgd_random == False:
                np.random.seed(0) # do not randomize
        self._v = verbose
        self.max_iters = max_iters
        self.nonZero = ratings > 0 # actual values
        
        print()
        if self.learning == 'als':
            print('ALS instance parameters:\nn_factors=%d, user_reg=%.5f,  item_reg=%.5f, num_iters=%d' %\
              (self.n_factors, self.user_fact_reg, self.item_fact_reg, self.max_iters))
        
        elif self.learning == 'sgd':
            print('SGD instance parameters:\nnum_factors K=%d, learn_rate alpha=%.5f, reg beta=%.5f, num_iters=%d, sgd_random=%s' %\
              (self.n_factors, self.sgd_alpha, self.sgd_beta, self.max_iters, self.sgd_random ) )
        print()

    def mf_train(self, n_iter=10): 
        """ Train model for n_iter iterations from scratch."""
        
        def normalize_row(x):
            norm_row =  x / sum(x) # weighted values: each row adds up to 1
            return norm_row

        # initialize latent vectors        
        self.user_vecs = np.random.normal(scale=1./self.n_factors,\
                                          size=(self.n_users, self.n_factors))
        self.item_vecs = np.random.normal(scale=1./self.n_factors,
                                          size=(self.n_items, self.n_factors))
        
        if self.learning == 'als':
            ## Try one of these. apply_long_axis came from Explicit_RS_MF_ALS()
            ##                                             Daniel Nee code
            
            self.user_vecs = abs(np.random.randn(self.n_users, self.n_factors))
            self.item_vecs = abs(np.random.randn(self.n_items, self.n_factors))
            
            #self.user_vecs = np.apply_along_axis(normalize_row, 1, self.user_vecs) # axis=1, across rows
            #self.item_vecs = np.apply_along_axis(normalize_row, 1, self.item_vecs) # axis=1, across rows
            
            self.partial_train(n_iter)
            
        elif self.learning == 'sgd':
            self.user_bias = np.zeros(self.n_users)
            self.item_bias = np.zeros(self.n_items)
            self.global_bias = np.mean(self.ratings[np.where(self.ratings != 0)])
            self.partial_train(n_iter)
    
    
    def partial_train(self, n_iter):
        """ 
        Train model for n_iter iterations. 
        Can be called multiple times for further training.
        Remains in the while loop for a number of iterations, calculated from
        the contents of the iter_array in calculate_learning_curve()
        """
        
        ctr = 1
        while ctr <= n_iter:

            if self.learning == 'als':
                self.user_vecs = self.als_step(self.user_vecs, 
                                               self.item_vecs, 
                                               self.ratings, 
                                               self.user_fact_reg, 
                                               type='user')
                self.item_vecs = self.als_step(self.item_vecs, 
                                               self.user_vecs, 
                                               self.ratings, 
                                               self.item_fact_reg, 
                                               type='item')
                
            elif self.learning == 'sgd':
                self.training_indices = np.arange(self.n_samples)
                np.random.shuffle(self.training_indices)
                self.sgd()
            ctr += 1

    def als_step(self,
                 latent_vectors,
                 fixed_vecs,
                 ratings,
                 _lambda,
                 type='user'):
        """
        ALS algo step.
        Solve for the latent vectors specified by type parameter: user or item
        """
        
        #lv_shape = latent_vectors.shape[0] ## debug
        
        if type == 'user':

            for u in range(latent_vectors.shape[0]): # latent_vecs ==> user_vecs
                #r_u = ratings[u, :] ## debug
                #fvT = fixed_vecs.T ## debug
                idx = self.nonZero[u,:] # get the uth user profile with booleans 
                                        # (True when there are ratings) based on 
                                        # ratingsMatrix, n x 1
                nz_fixed_vecs = fixed_vecs[idx,] # get the item vector entries, non-zero's x f
                YTY = nz_fixed_vecs.T.dot(nz_fixed_vecs) # fixed_vecs are item_vecs
                lambdaI = np.eye(YTY.shape[0]) * _lambda
                
                latent_vectors[u, :] = \
                    solve( (YTY + lambdaI) , nz_fixed_vecs.T.dot (ratings[u, idx] ) )

                '''
                ## debug
                if u <= 10: 
                    print('user vecs1', nz_fixed_vecs)
                    print('user vecs1', fixed_vecs, '\n', ratings[u, :] )
                    print('user vecs2', fixed_vecs.T.dot (ratings[u, :] ))
                    print('reg', YTY, '\n', lambdaI)
                    print('new user vecs:\n', latent_vectors[u, :])
                ## debug
                '''
                    
        elif type == 'item':
            
            for i in range(latent_vectors.shape[0]): #latent_vecs ==> item_vecs
                idx = self.nonZero[:,i] # get the ith item "profile" with booleans 
                                        # (True when there are ratings) based on 
                                        # ratingsMatrix, n x 1
                nz_fixed_vecs = fixed_vecs[idx,] # get the item vector entries, non-zero's x f
                XTX = nz_fixed_vecs.T.dot(nz_fixed_vecs) # fixed_vecs are user_vecs
                lambdaI = np.eye(XTX.shape[0]) * _lambda
                latent_vectors[i, :] = \
                    solve( (XTX + lambdaI) , nz_fixed_vecs.T.dot (ratings[idx, i] ) )

        return latent_vectors

    def sgd(self):
        ''' run sgd algo '''
        
        for idx in self.training_indices:
            u = self.sample_row[idx]
            i = self.sample_col[idx]
            prediction = self.predict(u, i)
            e = (self.ratings[u,i] - prediction) # error
            
            # Update biases
            self.user_bias[u] += self.sgd_alpha * \
                                (e - self.sgd_beta * self.user_bias[u])
            self.item_bias[i] += self.sgd_alpha * \
                                (e - self.sgd_beta * self.item_bias[i])
            
            # Create copy of row of user_vecs since we need to update it but
            #    use older values for update on item_vecs, 
            #    so make a deepcopy of previous user_vecs
            previous_user_vecs = deepcopy(self.user_vecs[u, :])
            
            # Update latent factors
            self.user_vecs[u, :] += self.sgd_alpha * \
                                    (e * self.item_vecs[i, :] - \
                                     self.sgd_beta * self.user_vecs[u,:])
            self.item_vecs[i, :] += self.sgd_alpha * \
                                    (e * previous_user_vecs - \
                                     self.sgd_beta * self.item_vecs[i,:])           
    
    def calculate_learning_curve(self, iter_array, test):
        """
        Keep track of MSE as a function of training iterations.
        
        Params
        ======
        iter_array : (list)
            List of numbers of iterations to train for each step of 
            the learning curve. e.g. [1, 5, 10, 20]
        test : (2D ndarray)
            Testing dataset (assumed to be user x item)
        
        
        
        This function creates two new class attributes:
        
        train_mse : (list)
            Training data MSE values for each value of iter_array
        test_mse : (list)
            Test data MSE values for each value of iter_array
        """
        
        print()
        if self.learning == 'als':
            print('Runtime parameters:\nn_factors=%d, user_reg=%.5f, item_reg=%.5f,'
                  ' max_iters=%d,'
                  ' \nratings matrix: %d users X %d items' %\
                  (self.n_factors, self.user_fact_reg, self.item_fact_reg, 
                   self.max_iters, self.n_users, self.n_items))
        if self.learning == 'sgd':
            print('Runtime parameters:\nn_factors=%d, learning_rate alpha=%.3f,'
                  ' reg beta=%.5f, max_iters=%d, sgd_random=%s'
                  ' \nratings matrix: %d users X %d items' %\
                  (self.n_factors, self.sgd_alpha, self.sgd_beta, 
                   self.max_iters, self.sgd_random, self.n_users, self.n_items))
        print()       
        
        iter_array.sort()
        self.train_mse =[]
        self.test_mse = []
        iter_diff = 0
        mse_iters = []
        start_time = time()
        stop_time = time()
        elapsed_time = (stop_time-start_time) #/60
        print ( 'Elapsed train/test time %.2f secs' % elapsed_time )        
        
        # Loop through number of iterations
        for (i, n_iter) in enumerate(iter_array):
            if self._v:
                print ('Iteration: {}'.format(n_iter))
            if i == 0:
                self.mf_train(n_iter - iter_diff) # init training, run first iter
            else:
                self.partial_train(n_iter - iter_diff) # run more iterations
                    # .. as you go from one element of iter_array to another

            predictions = self.predict_all() # calc dot product of p and qT
            # calc train  errors -- predicted vs actual
            self.train_mse += [self.get_mse(predictions, self.ratings)]
            if test.any() > 0: # check if test matrix is all zeroes ==> Train Only
                               # If so, do not calc mse and avoid runtime error   
                # calc test errors -- predicted vs actual 
                self.test_mse += [self.get_mse(predictions, test)]
            else:
                self.test_mse = ['n/a']
            if self._v:
                print ('Train mse: ' + str(self.train_mse[-1]))
                if self.test_mse != ['n/a']:
                    print ('Test mse: ' + str(self.test_mse[-1]))
            iter_diff = n_iter
            
            stop_time = time()
            elapsed_time = (stop_time-start_time) #/60
            print ( 'Elapsed train/test time %.2f secs' % elapsed_time ) 
            
        return self.test_mse, self.train_mse

           

    def predict(self, u, i):
        """ Single user and item prediction """
        
        if self.learning == 'als':
            prediction = self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
            if prediction > 5:
                prediction = 5.0 
            elif prediction < 0: 
                prediction = 1.0
            return prediction
        elif self.learning == 'sgd':
            prediction = self.global_bias + self.user_bias[u] + self.item_bias[i]
            prediction += self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
            if prediction > 5:
                prediction = 5.0 
            elif prediction < 0: 
                prediction = 1.0
            return prediction
    
    def predict_all(self):
        """ Predict ratings for every user and item """
        
        predictions = np.zeros((self.user_vecs.shape[0], 
                                self.item_vecs.shape[0]))
        for u in range(self.user_vecs.shape[0]):
            for i in range(self.item_vecs.shape[0]):
                predictions[u, i] = self.predict(u, i)
        return predictions    

    def get_mse(self, pred, actual):
        ''' Calc MSE between predicted and actual values '''
        
        # Ignore nonzero terms.
        pred = pred[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()
        return mean_squared_error(pred, actual)

def ratings_to_2D_matrix(ratings, m, n):
    '''
    creates a U-I matrix from the data
    ==>>  eliminates movies (items) that have no ratings!
    '''
    print('Summary Stats:')
    print()
    print(ratings.describe())
    ratingsMatrix = ratings.pivot_table(columns=['item_id'], index =['user_id'],
        values='rating', dropna = False) # convert to a U-I matrix format from file input
    ratingsMatrix = ratingsMatrix.fillna(0).values # replace nan's with zeroes
    ratingsMatrix = ratingsMatrix[0:m,0:n] # get rid of any users/items that have no ratings
    print()
    print('2D_matrix shape', ratingsMatrix.shape) # debug
    
    return ratingsMatrix

def file_info(df):
    ''' print file info/stats  '''
    print()
    print (df.head())
    
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    
    ratings = ratings_to_2D_matrix(df, n_users, n_items)
    
    print()
    print (ratings)
    print()
    print (str(n_users) + ' users')
    print (str(n_items) + ' items')
    
    sparsity = float(len(ratings.nonzero()[0]))
    sparsity /= (ratings.shape[0] * ratings.shape[1])
    sparsity *= 100
    sparsity = 100 - sparsity
    print ('Sparsity: {:4.2f}%'.format(sparsity))
    return ratings


    lcvs

def test_train_info(test, train):
    ''' print test/train info   '''

    print()
    print ('Train info: %d rows, %d cols' % (len(train), len(train[0])))
    print ('Test info: %d rows, %d cols' % (len(test), len(test[0])))
    
    test_count = 0
    for i in range(len(test)):
        for j in range(len(test[0])):
            if test[i][j] !=0:
                test_count += 1
                #print (i,j,test[i][j]) # debug
    print('test ratings count =', test_count)
    
    train_count = 0
    for i in range(len(train)):
        for j in range(len(train[0])):
            if train[i][j] !=0:
                train_count += 1
                #print (i,j,train[i][j]) # debug
    
    total_count = test_count + train_count
    print('train ratings count =', train_count)
    print('test + train count', total_count)
    print('test/train percentages: %0.2f / %0.2f' 
          % ( (test_count/total_count)*100, (train_count/total_count)*100 ))
    print()

def plot_learning_curve(iter_array, model):
    ''' plot the error curve '''
    
    ## Note: the iter_array can cause plots to NOT 
    ##    be smooth! If matplotlib can't smooth, 
    ##    then print/plot results every 
    ##    max_num_iterations/10 (rounded up)
    ##    instead of using an iter_array list
    
    #print('model.test_mse', model.test_mse) # debug
    if model.test_mse != ['n/a']:
        plt.plot(iter_array, model.test_mse, label='Test', linewidth=3)
    plt.plot(iter_array, model.train_mse, label='Train', linewidth=3)

    plt.xticks(fontsize=10); # 16
    plt.xticks(iter_array, iter_array)
    plt.yticks(fontsize=10);
    
    axes = plt.gca()
    axes.grid(True) # turns on grid
    
    if model.learning == 'als':
        runtime_parms = \
            'shape=%s, n_factors=%d, user_fact_reg=%.3f, item_fact_reg=%.3f'%\
            (model.ratings.shape, model.n_factors, model.user_fact_reg, model.item_fact_reg)
            #(train.shape, model.n_factors, model.user_fact_reg, model.item_fact_reg)
        plt.title("ALS Model Evaluation\n%s" % runtime_parms , fontsize=10) 
    elif model.learning == 'sgd':
        runtime_parms = \
            'shape=%s, num_factors K=%d, alpha=%.3f, beta=%.3f'%\
            (model.ratings.shape, model.n_factors, model.sgd_alpha, model.sgd_beta)
            #(train.shape, model.n_factors, model.learning_rate, model.user_fact_reg)
        plt.title("SGD Model Evaluation\n%s" % runtime_parms , fontsize=10)         
    
    plt.xlabel('Iterations', fontsize=15);
    plt.ylabel('Mean Squared Error', fontsize=15);
    plt.legend(loc='best', fontsize=15, shadow=True) # 'best', 'center right' 20
    
def ncf_read_data(filename):
    header = ['user_id','item_id','rating','timestamp']
    dataset = pd.read_csv(filename,sep = '\t',names = header)
    print(dataset.head())
    return dataset

def ncf_evaluate_recs(predictions, test, user): 

    mse_error_list, mae_error_list = [], []
    for i in range(len(predictions)): 
        pred = predictions[i]
        actual = test.rating.iloc[i]
        se = (pred - actual)**2
        error = abs(pred-actual)
        mse_error_list.append(se)
        mae_error_list.append(error)

    mse = sum(mse_error_list)/len(mse_error_list)
    mae = sum(mae_error_list)/len(mae_error_list)
    rmse = math.sqrt(mse)
    print("Test MSE for user %d = %f " % (user, mse))
    print("Test MAE for user %d = %f " % (user, mae))
    print("Test RMSE for user %d = %f " % (user, rmse))
    
    return mse, mae, rmse, len(mse_error_list)

def from_file_to_dict(path, datafile, itemfile):
    ''' Load user-item matrix from specified file 
        
        Parameters:
        -- path: directory path to datafile and itemfile
        -- datafile: delimited file containing userid, itemid, rating
        -- itemfile: delimited file that maps itemid to item name
        
        Returns:
        -- prefs: a nested dictionary containing item ratings for each user
    
    '''
    
    # Get movie titles, place into movies dictionary indexed by itemID
    movies={}
    try:
        with open (path + '/' + itemfile, encoding='iso8859') as myfile: 
            # this encoding is required for some datasets: encoding='iso8859'
            for line in myfile:
                (id,title)=line.split('|')[0:2]
                movies[id]=title.strip()
    
    # Error processing
    except UnicodeDecodeError as ex:
        print (ex)
        print (len(movies), line, id, title)
        return {}
    except Exception as ex:
        print (ex)
        print (len(movies))
        return {}
    
    # Load data into a nested dictionary
    prefs={}
    for line in open(path+'/'+ datafile, encoding='iso8859'):
        #print(line, line.split('\t')) #debug
        (user,movieid,rating,ts)=line.split('\t')
        user = user.strip() # remove spaces
        movieid = movieid.strip() # remove spaces
        prefs.setdefault(user,{}) # make it a nested dicitonary
        prefs[user][movies[movieid]]=float(rating)
    
    #return a dictionary of preferences
    return prefs

def file_info(df):
    ''' print file info/stats  '''
    print()
    print (df.head())
    
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    
    ratings = ratings_to_2D_matrix(df, n_users, n_items)
    
    print()
    print (ratings)
    print()
    print (str(n_users) + ' users')
    print (str(n_items) + ' items')
    
    sparsity = float(len(ratings.nonzero()[0]))
    sparsity /= (ratings.shape[0] * ratings.shape[1])
    sparsity *= 100
    sparsity = 100 - sparsity
    print ('Sparsity: {:4.2f}%'.format(sparsity))
    return ratings

def ratings_to_2D_matrix(ratings, m, n):
    '''
    creates a U-I matrix from the data
    ==>>  eliminates movies (items) that have no ratings!
    '''
    print('Summary Stats:')
    print()
    print(ratings.describe())
    ratingsMatrix = ratings.pivot_table(columns=['item_id'], index =['user_id'],
        values='rating', dropna = False) # convert to a U-I matrix format from file input
    ratingsMatrix = ratingsMatrix.fillna(0).values # replace nan's with zeroes
    ratingsMatrix = ratingsMatrix[0:m,0:n] # get rid of any users/items that have no ratings
    print()
    print('2D_matrix shape', ratingsMatrix.shape) # debug
    
    return ratingsMatrix

def mf_train_test_split(ratings, TRAIN_ONLY):
    ''' split the data into train and test '''
    test = np.zeros(ratings.shape)
    train = deepcopy(ratings) # instead of copy()
    
    ## setting the size parameter for random.choice() based on dataset size
    if len(ratings) < 10: # critics
        size = 1
    elif len(ratings) < 1000: # ml-100k
        size = 20
    else:
        size = 40 # ml-1m
        
    #print('size =', size) ## debug
    
    if TRAIN_ONLY == False:
        np.random.seed(0) # do not randomize the random.choice() in this function,
                          # let ALS or SGD make the decision to randomize
                          # Note: this decision can be reset with np.random.seed()
                          # .. see code at the end of this for loop
        for user in range(ratings.shape[0]): ## CES changed all xrange to range for Python v3
            test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                            size=size, 
                                            replace=True) #False)
            # When replace=False, size for ml-100k = 20, for critics = 1,2, or 3
            # Use replace=True for "better" results
            
            '''
            np.random.choice() info ..
            https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
            
            random.choice(a, size=None, replace=True, p=None)
            
            Parameters --
            a:         1-D array-like or int
            If an ndarray, a random sample is generated from its elements. 
            If an int, the random sample is generated as if it were np.arange(a)
            
            size:      int or tuple of ints, optional
            Output shape. If the given shape is, e.g., (m, n, k), 
            then m * n * k samples are drawn. 
            Default is None, in which case a single value is returned.
        
            replace:   boolean, optional
            Whether the sample is with or without replacement. 
            Default is True, meaning that a value of a can be selected multiple times.
        
            p:        1-D array-like, optional
            The probabilities associated with each entry in a. If not given, 
            the sample assumes a uniform distribution over all entries in a.
    
            Returns
            samples:   single item or ndarray
            The generated random samples
            
            '''
            
            train[user, test_ratings] = 0.
            test[user, test_ratings] = ratings[user, test_ratings]
            
        # Test and training are truly disjoint
        assert(np.all((train * test) == 0)) 
        np.random.seed() # allow other functions to randomize
    
    #print('TRAIN_ONLY (in split) =', TRAIN_ONLY) ##debug
    
    return train, test
    
def test_train_info(test, train):
    ''' print test/train info   '''

    print()
    print ('Train info: %d rows, %d cols' % (len(train), len(train[0])))
    print ('Test info: %d rows, %d cols' % (len(test), len(test[0])))
    
    test_count = 0
    for i in range(len(test)):
        for j in range(len(test[0])):
            if test[i][j] !=0:
                test_count += 1
                #print (i,j,test[i][j]) # debug
    print('test ratings count =', test_count)
    
    train_count = 0
    for i in range(len(train)):
        for j in range(len(train[0])):
            if train[i][j] !=0:
                train_count += 1
                #print (i,j,train[i][j]) # debug
    
    total_count = test_count + train_count
    print('train ratings count =', train_count)
    print('test + train count', total_count)
    print('test/train percentages: %0.2f / %0.2f' 
          % ( (test_count/total_count)*100, (train_count/total_count)*100 ))
    print()

def data_stats(prefs, filename):
    ''' Computes/prints descriptive analytics:
        -- Total number of users, items, ratings
        -- Overall average rating, standard dev (all users, all items)
        -- Average item rating, standard dev (all users)
        -- Average user rating, standard dev (all items)
        -- Matrix ratings sparsity
        -- Ratings distribution histogram (all users, all items)

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- filename: string containing name of file being analyzed
        
        Returns:
        -- None

    '''
    num_users = len(prefs)
    
    ratings = []
    item_ratings = {}
    
    for user in prefs.keys(): 
        user_ratings = prefs[user]
        for item in user_ratings.keys():
            if item in item_ratings:
                item_ratings[item].append(user_ratings[item])
            else: 
                item_ratings[item] = [user_ratings[item]]
            ratings.append(user_ratings[item])
        
    num_items = len(item_ratings)
    num_ratings = len(ratings)
    
    overall_avg_rating = sum(ratings)/num_ratings
    overall_std = np.std(ratings)

    item_avgs = {}
    for item in item_ratings.keys():
        item_avgs[item] = sum(item_ratings[item])/len(item_ratings[item])
        
    item_avg_rating = sum([item_avgs[item] for item in item_avgs.keys()])/num_items
    item_std = np.std([item_avgs[item] for item in item_avgs.keys()])

    user_avgs = {}
    user_num_ratings = {}
    for user in prefs.keys():
        user_num_ratings[user] = len([prefs[user][item] for item in prefs[user]])
        user_avgs[user] = sum([prefs[user][item] for item in prefs[user]])/len(prefs[user])

    num_ratings_list = [user_num_ratings[user] for user in prefs.keys()]
    avg_num_user_ratings = np.mean(num_ratings_list)
    std_dev_user_ratings = np.std(num_ratings_list)
    min_user_ratings = np.min(num_ratings_list)
    max_user_ratings = np.max(num_ratings_list) 
    median_user_ratings = np.median(num_ratings_list)

    user_avg_rating = sum([user_avgs[user] for user in user_avgs.keys()])/num_users
    user_std = np.std([user_avgs[user] for user in user_avgs.keys()])

    sparsity = (1 - (num_ratings/(num_users*num_items))) * 100

    print("Number of users: %d" % num_users)
    print("Number of items: %d" % num_items)
    print("Number of ratings: %d"% num_ratings)
    print("Average number of ratings per user: %.2f" % avg_num_user_ratings)
    print("Standard dev of ratings per user: %.2f" % std_dev_user_ratings)
    print("Min number of ratings per user: %.2f" % min_user_ratings)
    print("Max number of ratings per user: %.2f" % max_user_ratings)
    print("Median number of ratings per user: %.2f" % median_user_ratings)
    print("Overall average rating: %.2f out of 5, and std dev of %.2f" % (overall_avg_rating, overall_std))
    # print("Item Avg Ratings List: ", [item_avgs[item] for item in item_avgs.keys()])
    print("Average item rating: %.2f out of 5, and std dev of %.2f" % (item_avg_rating, item_std))
    # print("User Ratings List: ", [user_avgs[user] for user in prefs.keys()])
    print("Average user rating: %.2f out of 5, and std dev of %.2f" % (user_avg_rating, user_std))
    print("User-Item Matrix Sparsity: %.2f" % sparsity, '%')


    plt.hist(ratings, 4, facecolor='blue', alpha = 0.75)
    plt.title("Ratings Histogram")
    #plt.xticks(np.arange(1,6, step=1))
    #plt.yticks(np.arange(0, 19, step=2))
    plt.xlabel('Rating')
    plt.ylabel('Number of user ratings')
    plt.grid()
    plt.show()
    plt.close()

def from_file_to_2D(path, genrefile, itemfile):
    ''' Load feature matrix from specified file 
        Parameters:
        -- path: directory path to datafile and itemfile
        -- genrefile: delimited file that maps genre to genre index
        -- itemfile: delimited file that maps itemid to item name and genre
        
        Returns:
        -- movies: a dictionary containing movie titles (value) for a given movieID (key)
        -- genres: dictionary, key is genre, value is index into row of features array
        -- features: a 2D list of features by item, values are 1 and 0;
                     rows map to items and columns map to genre
                     returns as np.array()
    
    '''
    # Get movie titles, place into movies dictionary indexed by itemID
    movies={}
    try:
        with open (path + '/' + itemfile, encoding='iso8859') as myfile: 
            # this encoding is required for some datasets: encoding='iso8859'
            for line in myfile:
                (id,title)=line.split('|')[0:2] 
                movies[id]=title.strip()
    
    # Error processing
    except UnicodeDecodeError as ex:
        print (ex)
        print (len(movies), line, id, title)
        return {}
    except ValueError as ex:
        print ('ValueError', ex)
        print (len(movies), line, id, title)
    except Exception as ex:
        print (ex)
        print (len(movies))
        return {}
    
    ##
    # Get movie genre from the genre file, place into genre dictionary indexed by genre index
    genres={} # key is genre index, value is the genre string
    try: 
        for line in open(path+'/'+ genrefile, encoding='iso8859'):
            #print(line, line.split('|')) #debug
            fields = line.split('|')
            genres[int(fields[1].split('\n')[0])] = fields[0]
    except Exception as ex:
        print (ex)
        print ('Proceeding with len(genres)', len(genres))
    


    
    # Load data into a nested 2D list
    features = []
    start_feature_index = 5
    try: 
        for line in open(path+'/'+ itemfile, encoding='iso8859'):
            #print(line, line.split('|')) #debug
            fields = line.split('|')[start_feature_index:]
            row = []
            for feature in fields:
                row.append(int(feature))
            features.append(row)
        features = np.array(features)
    except Exception as ex:
        print (ex)
        print ('Proceeding with len(features)', len(features))
        #return {}
    
    #return features matrix
    return movies, genres, features  

def popular_items(prefs, filename): 
    ''' Computes/prints popular items analytics    
        -- popular items: most rated (sorted by # ratings)
        -- popular items: highest rated (sorted by avg rating)
        -- popular items: highest rated items that have at least a 
                          "threshold" number of ratings
        
        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- filename: string containing name of file being analyzed
        
        Returns:
        -- None

    '''
    items_dict = {}
    
    for i in prefs:
        for j in prefs[i].keys():
            items_dict[j] = [0,0,0]
            
    for i in prefs:
        for j in prefs[i].keys():
            items_dict[j][0] += prefs[i][j]
            items_dict[j][1] += 1
    
    for i in items_dict:
        items_dict[i][2] = items_dict[i][0] / items_dict[i][1]
    
    

    items_df_1 = pd.DataFrame(items_dict)
    items_df = items_df_1.transpose()
    
    ratings_sorted_items_df = items_df.sort_values(by=[1], ascending=False)
    
    print('\nPopular items -- most rated: \nTitle \t\t\t\t #Ratings \t Avg Rating \n')
    print('%s \t\t %d \t\t %.2f' % (ratings_sorted_items_df.index[0],ratings_sorted_items_df.iloc[0][1], ratings_sorted_items_df.iloc[0][2]))
    print('%s \t\t\t %d \t\t %.2f' % (ratings_sorted_items_df.index[1],ratings_sorted_items_df.iloc[1][1], ratings_sorted_items_df.iloc[1][2]))
    print('%s \t\t\t %d \t\t %.2f' % (ratings_sorted_items_df.index[2],ratings_sorted_items_df.iloc[2][1], ratings_sorted_items_df.iloc[2][2]))
    print('%s \t %d \t\t %.2f' % (ratings_sorted_items_df.index[3],ratings_sorted_items_df.iloc[3][1], ratings_sorted_items_df.iloc[3][2]))
    print('%s \t\t %d \t\t %.2f' % (ratings_sorted_items_df.index[4],ratings_sorted_items_df.iloc[4][1], ratings_sorted_items_df.iloc[4][2]))
    
    avg_ratings_sorted_items_df = items_df.sort_values(by=[2], ascending=False)
    
    print('\nPopular items -- highest rated: \nTitle \t\t\t\t Avg Rating \t #Ratings \n')
    print('%s \t %.2f \t\t %d' % (avg_ratings_sorted_items_df.index[0],avg_ratings_sorted_items_df.iloc[0][2], avg_ratings_sorted_items_df.iloc[0][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df.index[1],avg_ratings_sorted_items_df.iloc[1][2], avg_ratings_sorted_items_df.iloc[1][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df.index[2],avg_ratings_sorted_items_df.iloc[2][2], avg_ratings_sorted_items_df.iloc[2][1]))
    print('%s \t %.2f \t\t %d' % (avg_ratings_sorted_items_df.index[3],avg_ratings_sorted_items_df.iloc[3][2], avg_ratings_sorted_items_df.iloc[3][1]))
    print('%s \t %.2f \t\t %d' % (avg_ratings_sorted_items_df.index[4],avg_ratings_sorted_items_df.iloc[4][2], avg_ratings_sorted_items_df.iloc[4][1]))
    
    items_df_min = items_df[items_df[1] >= 20]
    avg_ratings_sorted_items_df_min = items_df_min.sort_values(by=[2], ascending=False)
    
    print('\nOverall best rated items (number of ratings >= 20): \nTitle \t\t\t\t Avg Rating \t #Ratings \n')
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[0],avg_ratings_sorted_items_df_min.iloc[0][2], avg_ratings_sorted_items_df_min.iloc[0][1]))
    print('%s \t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[1],avg_ratings_sorted_items_df_min.iloc[1][2], avg_ratings_sorted_items_df_min.iloc[1][1]))
    print('%s \t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[2],avg_ratings_sorted_items_df_min.iloc[2][2], avg_ratings_sorted_items_df_min.iloc[2][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[3],avg_ratings_sorted_items_df_min.iloc[3][2], avg_ratings_sorted_items_df_min.iloc[3][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[4],avg_ratings_sorted_items_df_min.iloc[4][2], avg_ratings_sorted_items_df_min.iloc[4][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[5],avg_ratings_sorted_items_df_min.iloc[5][2], avg_ratings_sorted_items_df_min.iloc[5][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[6],avg_ratings_sorted_items_df_min.iloc[6][2], avg_ratings_sorted_items_df_min.iloc[6][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[7],avg_ratings_sorted_items_df_min.iloc[7][2], avg_ratings_sorted_items_df_min.iloc[7][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[8],avg_ratings_sorted_items_df_min.iloc[8][2], avg_ratings_sorted_items_df_min.iloc[8][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[9],avg_ratings_sorted_items_df_min.iloc[9][2], avg_ratings_sorted_items_df_min.iloc[9][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[10],avg_ratings_sorted_items_df_min.iloc[10][2], avg_ratings_sorted_items_df_min.iloc[10][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[11],avg_ratings_sorted_items_df_min.iloc[11][2], avg_ratings_sorted_items_df_min.iloc[11][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[12],avg_ratings_sorted_items_df_min.iloc[12][2], avg_ratings_sorted_items_df_min.iloc[12][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[13],avg_ratings_sorted_items_df_min.iloc[13][2], avg_ratings_sorted_items_df_min.iloc[13][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[14],avg_ratings_sorted_items_df_min.iloc[14][2], avg_ratings_sorted_items_df_min.iloc[14][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[15],avg_ratings_sorted_items_df_min.iloc[15][2], avg_ratings_sorted_items_df_min.iloc[15][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[16],avg_ratings_sorted_items_df_min.iloc[16][2], avg_ratings_sorted_items_df_min.iloc[16][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[17],avg_ratings_sorted_items_df_min.iloc[17][2], avg_ratings_sorted_items_df_min.iloc[17][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[18],avg_ratings_sorted_items_df_min.iloc[18][2], avg_ratings_sorted_items_df_min.iloc[18][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[19],avg_ratings_sorted_items_df_min.iloc[19][2], avg_ratings_sorted_items_df_min.iloc[19][1]))

def sim_pearson(prefs,p1,p2, weight):
    '''
        Calculate Pearson Correlation similarity 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2
        
        Returns:
        -- Pearson Correlation similarity as a float
        
    Source: Programming Collective Intelligence, Segaran 2007
    '''
    # Get the list of mutually rated items
    si={}
    for item in prefs[p1]: 
        if item in prefs[p2]: 
            si[item]=1
  
    # if there are no ratings in common, return 0
    if len(si)==0: 
        return 0
  
    # Sum calculations
    n=len(si)
    
    
    # Sums of all the preferences
    sum1=sum([prefs[p1][it] for it in si])
    sum2=sum([prefs[p2][it] for it in si])
    
    # Sums of the squares
    sum1Sq=sum([pow(prefs[p1][it],2) for it in si])
    sum2Sq=sum([pow(prefs[p2][it],2) for it in si])   
    
    # Sum of the products
    pSum=sum([prefs[p1][it]*prefs[p2][it] for it in si])
    
    # Calculate r (Pearson score)
    num=pSum-(sum1*sum2/n)
    den=sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
    if den==0: 
        return 0
    
    if n < weight and weight != 1:
        r=(num/den)*(n/weight)
    else: 
        r = (num/den)
    
  
    return r

def sim_distance(prefs,person1,person2, weight):
    '''
        Calculate Euclidean distance similarity 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2
        
        Returns:
        -- Euclidean distance similarity as a float
    
    Source: Programming Collective Intelligence, Segaran 2007
    '''
    
    # Get the list of shared_items
    si={}
    for item in prefs[person1]: 
        if item in prefs[person2]: 
            si[item]=1
    
    # if they have no ratings in common, return 0
    if len(si)==0: 
        return 0
    
    n = len(si)
    
    # Add up the squares of all the differences
    sum_of_squares=sum([pow(prefs[person1][item]-prefs[person2][item],2) 
                        for item in prefs[person1] if item in prefs[person2]])
    
    '''
    ## FYI, This is what the list comprehension above breaks down to ..
    ##
    sum_of_squares = 0
    for item in prefs[person1]:
        if item in prefs[person2]:
            #print(item, prefs[person1][item], prefs[person2][item])
            sq = pow(prefs[person1][item]-prefs[person2][item],2)
            #print (sq)
            sum_of_squares += sq
    '''
    
    sim = 1/(1+sqrt(sum_of_squares)) 
    if n < weight and weight != 1:
        sim = sim * (n/weight)

    return sim
    
def sim_cosine(prefs, p1, p2, weight):
    '''
    Info:
    https://www.geeksforgeeks.org/cosine-similarity/
    https://en.wikipedia.org/wiki/Cosine_similarity
    
    Source for some of the code: https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists

    '''
    # Get the list of mutually rated items
    si={}
    for item in prefs[p1]: 
        if item in prefs[p2]: 
            si[item]=1

    # if there are no ratings in common, return 0
    if len(si)==0: 
        return 0 
    n = len(si)
    # Sum of the products
    sumxy=([prefs[p1][it]*prefs[p2][it] for it in si])
    sumxx=([prefs[p1][it]*prefs[p1][it] for it in si])
    sumyy=([prefs[p2][it]*prefs[p2][it] for it in si])
    #print (sumxy, sumxx, sumyy)
    sumxy=sum(sumxy)
    sumxx=sum(sumxx)
    sumyy=sum(sumyy)
    
    # Calculate r (cosine sim score)
    num = sumxy
    den = sqrt(sumxx*sumyy) 
    
    if den==0: 
        return 0
    if n < weight and weight != 1:
        r=(num/den)*(n/weight)
    else: 
        r = (num/den)
    return r

def prefs_to_2D_list(prefs):
    '''
    Convert prefs dictionary into 2D list used as input for the MF class
    
    Parameters: 
        prefs: user-item matrix as a dicitonary (dictionary)
        
    Returns: 
        ui_matrix: (list) contains user-item matrix as a 2D list
        
    '''
    ui_matrix = []
    
    user_keys_list = list(prefs.keys())
    num_users = len(user_keys_list)
    #print (len(user_keys_list), user_keys_list[:10]) # debug
    
    itemPrefs = transformPrefs(prefs) # traspose the prefs u-i matrix
    item_keys_list = list(itemPrefs.keys())
    num_items = len(item_keys_list)
    #print (len(item_keys_list), item_keys_list[:10]) # debug
    
    sorted_list = True # <== set manually to test how this affects results
    
    if sorted_list == True:
        user_keys_list.sort()
        item_keys_list.sort()
        print ('\nsorted_list =', sorted_list)
        
    # initialize a 2D matrix as a list of zeroes with 
    #     num users (height) and num items (width)
    
    for i in range(num_users):
        row = []
        for j in range(num_items):
            row.append(0.0)
        ui_matrix.append(row)
          
    # populate 2D list from prefs
    # Load data into a nested list

    for user in prefs:
        for item in prefs[user]:
            user_idx = user_keys_list.index(user)
            movieid_idx = item_keys_list.index(item) 
            
            try: 
                # make it a nested list
                ui_matrix[user_idx][movieid_idx] = prefs [user][item] 
            except Exception as ex:
                print (ex)
                print (user_idx, movieid_idx)   
                
    # return 2D user-item matrix
    return ui_matrix

def to_array(prefs):
    ''' convert prefs dictionary into 2D list '''
    R = prefs_to_2D_list(prefs)
    R = np.array(R)
    print ('to_array -- height: %d, width: %d' % (len(R), len(R[0]) ) )
    return R

def to_string(features):
    ''' convert features np.array into list of feature strings '''
    
    feature_str = []
    for i in range(len(features)):
        row = ''
        for j in range(len (features[0])):
            row += (str(features[i][j]))
        feature_str.append(row)
    print ('to_string -- height: %d, width: %d' % (len(feature_str), len(feature_str[0]) ) )
    return feature_str

def to_docs(features_str, genres):
    ''' convert feature strings to a list of doc strings for TFIDF '''
    
    feature_docs = []
    for doc_str in features_str:
        row = ''
        for i in range(len(doc_str)):
            if doc_str[i] == '1':
                row += (genres[i] + ' ') # map the indices to the actual genre string
        feature_docs.append(row.strip()) # and remove that pesky space at the end
        
    print ('to_docs -- height: %d, width: varies' % (len(feature_docs) ) )
    return feature_docs

def cosine_sim(docs):
    ''' Perofmrs cosine sim calcs on features list, aka docs in TF-IDF world
    
        Parameters:
        -- docs: list of item features
     
        Returns:   
        -- list containing cosim_matrix: item_feature-item_feature cosine similarity matrix 
    
    
    '''
    
    print()
    print('## Cosine Similarity calc ##')
    print()
    print('Documents:', docs[:10])
    
    print()
    print ('## Count and Transform ##')
    print()
    
    # get the TFIDF vectors
    tfidf_vectorizer = TfidfVectorizer() # orig
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
    #print (tfidf_matrix.shape, type(tfidf_matrix)) # debug

    
    print()
    print('Document similarity matrix:')
    cosim_matrix = cosine_similarity(tfidf_matrix[0:], tfidf_matrix)
    print (type(cosim_matrix), len(cosim_matrix))
    print()
    print(cosim_matrix[0:6])
    print()
    
    '''
    print('Examples of similarity angles')
    if tfidf_matrix.shape[0] > 2:
        for i in range(6):
            cos_sim = cosim_matrix[1][i] #(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix))[0][i] 
            if cos_sim > 1: cos_sim = 1 # math precision creating problems!
            angle_in_radians = math.acos(cos_sim)
            print('Cosine sim: %.3f and angle between documents 2 and %d: ' 
                  % (cos_sim, i+1), end=' ')
            print ('%.3f degrees, %.3f radians' 
                   % (math.degrees(angle_in_radians), angle_in_radians))
    '''
    
    return cosim_matrix

def movie_to_ID(movies):
    ''' converts movies mapping from "id to title" to "title to id" '''

    mov_to_id = {}

    for id in movies:
        mov_to_id[movies[id]] = id

    return mov_to_id

def get_TFIDF_recommendations(prefs,cosim_matrix,user, thresh, movie_title_to_id):
    '''
        Calculates recommendations for a given user 
        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- cosim_matrix: list containing item_feature-item_feature cosine similarity matrix 
        -- user: string containing name of user requesting recommendation
        -- SIG_THRESHOLD: neighborhood similarity threshold
        -- movie_title_to_id: dictionary that maps movie title to movieid        
        Returns:
        -- predictions: A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        
    '''
    
    # find more details in Final Project Specification
    predictions = []
    items_to_rate = []

    for item, itemID in movie_title_to_id.items():
        if item not in prefs[user].keys():
            items_to_rate.append((item, int(itemID) - 1))
    
    for item,itemID in items_to_rate:
        den = 0
        num = 0

        for mov, rating in prefs[user].items():
            cossim = cosim_matrix[int(movie_title_to_id[mov])-1][itemID]
            if cossim > thresh:
                num += (cossim * rating)
                den += cossim
        
        if den != 0:
            predictions.append((num/den,item))
    
    predictions.sort(reverse = True)

    return predictions

def single_TFIDF_rec(prefs,cosim_matrix,user, thresh, movie_title_to_id, item):
        '''
        Calculates recommendations for a given user 
        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- cosim_matrix: list containing item_feature-item_feature cosine similarity matrix 
        -- user: string containing name of user requesting recommendation
        -- SIG_THRESHOLD: neighborhood similarity threshold
        -- movie_title_to_id: dictionary that maps movie title to movieid 
        -- item: item that requires recommendation for a specific user       
        Returns:
        -- ranknigs: A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        '''
        predictions = []
        items_to_rate = []

        for title, itemID in movie_title_to_id.items():
            if title == item:
                items_to_rate.append((item, int(itemID) - 1))
        
        for item,itemID in items_to_rate:
            den = 0
            num = 0

            for mov, rating in prefs[user].items():
                cossim = cosim_matrix[int(movie_title_to_id[mov])-1][itemID]
                if cossim > thresh:
                    num += (cossim * rating)
                    den += cossim
            
            if den != 0:
                predictions.append((num/den,item))
        
        predictions.sort(reverse = True)

        return predictions

def similarity_histogram(sim_matrix):

    n = np.shape(sim_matrix)[0]
    len_sims = n*(n-1)//2
    all_sims = []
    print(sim_matrix)

    for r in range(n):
        for c in range(r):
            sim = sim_matrix[r][c]
            if sim > TFIDF_SIG_THRESHOLD: 
                all_sims.append(sim)
       
    mean = np.mean(all_sims)
    std = np.std(all_sims)
    print("Mean = %f" % mean )
    print("STD = %f " % std)
    print(mean - std, mean+std)

    # 0, mean+-std, 0.1, 0.3 

    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print("Plotting Histogram...")
    plt.hist(all_sims, facecolor='blue', alpha = 0.75)
    plt.title("Ratings Histogram")
    #plt.xticks(np.arange(1,6, step=1))
    #plt.yticks(np.arange(0, 19, step=2))
    plt.xlabel('Similarity')
    plt.ylabel('Number of Instances')
    plt.grid()
    plt.show()

def to_array(prefs):
    ''' convert prefs dictionary into 2D list '''
    R = prefs_to_2D_list(prefs)
    R = np.array(R)
    print ('to_array -- height: %d, width: %d' % (len(R), len(R[0]) ) )
    return R 

def getRecommendations(prefs,person, similarity=sim_pearson):
    '''
        Calculates recommendations for a given user 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        
    '''
    
    totals={}
    simSums={}
    for other in prefs:
      # don't compare me to myself
        if other == person: 
            continue
        sim = similarity(prefs,person,other)
    
        # ignore scores of zero or lower
        if sim <= 0: continue
        for item in prefs[other]:
            
            # only score movies I haven't seen yet
            if item not in prefs[person] or prefs[person][item]==0:
                # Similarity * Score
                totals.setdefault(item,0)
                totals[item]+=prefs[other][item]*sim
                # Sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+=sim
  
    # Create the normalized list
    rankings=[(total/simSums[item],item) for item,total in totals.items()]
  
    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings

def getRecommendationsSim(prefs, person, usersim, weight, thresh):
    '''
        Calculates recommendations for a given user 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- usersim: previously calculated (within simu) user-user similarity matrix
        
        Returns:
        -- A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        
    '''
    totals={}
    simSums={}
    for other in prefs:
      # don't compare me to myself
        if other == person: 
            continue
        
        #finds the similarity weight from the user-user similarity matrix that has already been calculated
        # previously -- sim = similarity(prefs,person,other)
        sim = 0
        for tupl in usersim[person]:
            if tupl[1] == other:
                sim = tupl[0]
    
        # ignore scores of zero or lower
        if sim <= 0: continue
        if sim < thresh: continue 
        for item in prefs[other]:
            
            # only score movies I haven't seen yet
            if item not in prefs[person] or prefs[person][item]==0:
                # Similarity * Score
                totals.setdefault(item,0)
                totals[item]+=prefs[other][item]*sim
                # Sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+=sim
  
    # Create the normalized list
    rankings=[(total/simSums[item],item) for item,total in totals.items()]
  
    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings

def ext_getRecommendationsSim(prefs, person, usersim, removed_item, weight, thresh):
    '''
        Calculates recommendations for a given user 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- usersim: previously calculated (within simu) user-user similarity matrix
        
        Returns:
        -- A rankings list with only the item removed by the LOOCV process,
            or an empty list if no recommendation calculated.
        
    '''
    totals={}
    simSums={}
    for other in prefs:
      # don't compare me to myself
        if other == person: 
            continue
        
        # finds the similarity weight from the user-user similarity matrix that has already been calculated
        # previously -- sim = similarity(prefs,person,other)
        sim = 0
        for tupl in usersim[person]:
            if tupl[1] == other:
                sim = tupl[0]
        
    
        # ignore scores of zero or lower
        if sim <= 0: continue
        # threshold
        if sim < thresh: continue 

        for item in prefs[other]:
            
            # only score movies I haven't seen yet
            if item not in prefs[person] or prefs[person][item]==0:
                # Similarity * Score
                totals.setdefault(item,0)
                totals[item]+=prefs[other][item]*sim
                # Sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+=sim
  
    # Create the normalized list
    rankings=[(total/simSums[item],item) for item,total in totals.items()]
    single = []
    for tupl in rankings:
        if tupl[1] == removed_item:
            single.append(tupl)
  
    return single

def ext_getRecommendedItems(prefs, user, itemMatch, removed_item, weight, thresh):
    '''
        Calculates recommendations for a given user 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A rankings list with only the item removed by the LOOCV process,
            or an empty list if no recommendation calculated.
        
    '''    

    userRatings=prefs[user]
    scores={}
    totalSim={}
    # Loop over items rated by this user
    for (item,rating) in userRatings.items( ):
  
      # Loop over items similar to this one
        for (similarity,item2) in itemMatch[item]:
    
            # Ignore if this user has already rated this item
            if item2 in userRatings: continue
            # ignore scores of zero or lower
            if similarity<=0: continue 
            # test threshold           
            if similarity<thresh: continue 
            # Weighted sum of rating times similarity
            scores.setdefault(item2,0)
            scores[item2]+=similarity*rating
            # Sum of all the similarities
            totalSim.setdefault(item2,0)
            totalSim[item2]+=similarity
  
    # Divide each total score by total weighting to get an average

    rankings=[(score/totalSim[item],item) for item,score in scores.items( )]    
  
    single = []
    for tupl in rankings:
        if tupl[1] == removed_item:
            single.append(tupl)
  
    return single
    
def get_all_UU_recs(prefs, sim=sim_pearson, num_users=10, top_N=5):
    ''' 
    Print user-based CF recommendations for all users in dataset

    Parameters
    -- prefs: nested dictionary containing a U-I matrix
    -- sim: similarity function to use (default = sim_pearson)
    -- num_users: max number of users to print (default = 10)
    -- top_N: max number of recommendations to print per user (default = 5)

    Returns: None
    '''
    i = 0
    for user in prefs.keys(): 
        recs = getRecommendations(prefs, user, sim)
        print ('User-based CF recs for %s, sim_pearson: ' % (user), 
                       recs[:top_N]) 
        i += 1
        if i >= num_users: 
            break

def loo_cv(prefs, metric, sim, algo):
    """
    Leave_One_Out Evaluation: evaluates recommender system ACCURACY
     
     Parameters:
         prefs dataset: critics, ml-100K, etc.
         metric: MSE, MAE, RMSE, etc.
         sim: distance, pearson, etc.
         algo: user-based recommender, item-based recommender, etc.
     
    Returns:
         error_total: MSE, MAE, RMSE totals for this set of conditions
         error_list: list of actual-predicted differences
    
    
    Algo Pseudocode ..
    Create a temp copy of prefs
    
    For each user in temp copy of prefs:
      for each item in each user's profile:
          delete this item
          get recommendation (aka prediction) list
          restore this item
          if there is a recommendation for this item in the list returned
              calc error, save into error list
          otherwise, continue
      
    return mean error, error list

    """
    error_list = []
    for user in prefs.keys():
        for item in prefs[user].keys():
            actual = prefs[user][item]
            temp = copy.deepcopy(prefs)
            temp[user].pop(item) # remove item 
            recs = algo(temp, user, sim) # get recommendation
            # prefs[user][item] = actual_rating # restore item]
            for rec in recs: 
                if rec[1] == item: 
                    prediction = rec[0]
                    error = (actual - prediction)**2
                    error_list.append(error)
                    print("User: %s, Item: %s, Prediction: %.3f, Actual: %.3f, Sq Error: %.3f " % (user, item, prediction, actual, error))
            
            if item not in set([rec[1] for rec in recs]):
                print("No prediction/recommendation available for User: %s, Item: %s " % (user, item))
                continue
    
    
    error = np.average(error_list)

    print("MSE for critics: %.3f using %s " % (error, str(sim)))
    
    return error, error_list

def calc_user_dist_similarities(prefs):
    user_similarities = {}
    for user1 in prefs.keys():
        for user2 in prefs.keys():
            if user1 != user2: 
                user_similarities[(user1, user2)] = sim_distance(prefs, user1, user2)
                print('Distance sim %s & %s: %.3f' % (user1, user2, user_similarities[(user1, user2)]))
    
    return user_similarities 

def calc_user_pc_similarities(prefs):

    user_similarities = {}
    for user1 in prefs.keys():
        for user2 in prefs.keys():
            if user1 != user2: 
                user_similarities[(user1, user2)] = sim_pearson(prefs, user1, user2)
                print('Pearson sim %s & %s: %.3f' % (user1, user2, user_similarities[(user1, user2)]))
    
    return user_similarities 

def topMatches(prefs,person,weight,counter,similarity=sim_pearson, n=5):
    '''
        Returns the best matches for person from the prefs dictionary

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        -- n: number of matches to find/return (5 is default)
        
        Returns:
        -- A list of similar matches with 0 or more tuples, 
           each tuple contains (similarity, item name).
           List is sorted, high to low, by similarity.
           An empty list is returned when no matches have been calc'd.
        
    '''     
    scores=[(similarity(prefs,person,other, weight),other) 
                    for other in prefs if other!=person]
    scores.sort()
    scores.reverse()
    # if counter < 10:
    #     print("itemsim: %s" % person)
    #     print(scores[0:n])

    return scores[0:n]

def transformPrefs(prefs):
    '''
        Transposes U-I matrix (prefs dictionary) 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        
        Returns:
        -- A transposed U-I matrix, i.e., if prefs was a U-I matrix, 
           this function returns an I-U matrix
        
    '''     
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})
            # Flip item and person
            result[item][person]=prefs[person][item]
    return result

def calculateSimilarItems(prefs, weight, n=100,similarity=sim_pearson):
    '''
        Creates a dictionary of items showing which other items they are most 
        similar to. 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- n: number of similar matches for topMatches() to return
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A dictionary with a similarity matrix
        
    '''
    result={}
    # Invert the preference matrix to be item-centric
    itemPrefs=transformPrefs(prefs)
    c=0
    for item in itemPrefs:
      # Status updates for larger datasets
        c+=1
        if c%100==0: 
            print("%s / %d" % (c, len(itemPrefs)))
            
        # Find the most similar items to this one
        scores=topMatches(itemPrefs,item, weight, c, similarity,n=n)
        result[item]=scores
        
    return result

def calculateSimilarUsers(prefs, weight, n=100,similarity=sim_pearson):

    '''
        Creates a dictionary of items showing which other items they are most 
        similar to. 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- n: number of similar matches for topMatches() to return
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A dictionary with a similarity matrix
        
    '''     
    result={}
    c=0
    for user in prefs:
      # Status updates for larger datasets
        c+=1
        if c%100==0: 
            print("%d / %d" % (c, len(prefs)))
            
        # Find the most similar items to this one
        scores=topMatches(prefs,user,weight, c, similarity,n=n)
        result[user]=scores
    return result

def getRecommendedItems(prefs, user, itemMatch, weight, thresh):
    '''
        Calculates recommendations for a given user 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        
    '''    

    userRatings=prefs[user]
    scores={}
    totalSim={}
    # Loop over items rated by this user
    for (item,rating) in userRatings.items( ):
  
      # Loop over items similar to this one
        for (similarity,item2) in itemMatch[item]:
    
            # Ignore if this user has already rated this item
            if item2 in userRatings: continue
            # ignore scores of zero or lower
            if similarity<=0: continue            
            # Weighted sum of rating times similarity
            if similarity<thresh: continue 
            scores.setdefault(item2,0)
            scores[item2]+=similarity*rating*weight
            # Sum of all the similarities
            totalSim.setdefault(item2,0)
            totalSim[item2]+=similarity*weight
  
    # Divide each total score by total weighting to get an average

    rankings=[(score/totalSim[item],item) for item,score in scores.items( )]    
  
    # Return the rankings from highest to lowest
    rankings.sort( )
    rankings.reverse( )
    return rankings
           
def get_all_II_recs(prefs, itemsim, sim_method, num_users=10, top_N=5):
    ''' 
    Print item-based CF recommendations for all users in dataset

    Parameters
    -- prefs: U-I matrix (nested dictionary)
    -- itemsim: item-item similarity matrix (nested dictionary)
    -- sim_method: name of similarity method used to calc sim matrix (string)
    -- num_users: max number of users to print (integer, default = 10)
    -- top_N: max number of recommendations to print per user (integer, default = 5)

    Returns: None
    
    '''
    i = 0
    for user in prefs.keys(): 
        recs = getRecommendedItems(prefs, itemsim, user)
        print ('Item-based CF recs for %s, %s: ' % (user, sim_method), 
                       recs[:top_N]) 
        i += 1
        if i >= num_users: 
            break

def get_uu_cf_matrix(sim):

    if sim == "pearson": 
        usersim = pickle.load(open( "save_usersim_pearson.p", "rb" ))  
        
    elif sim == "distance":
        usersim = pickle.load(open( "save_usersim_distance.p", "rb" ))
    else: 
        print("User-based similarity matrix not computed, run SIMU command first")
        return None
    return usersim

def get_ii_cf_matrix(sim):

    if sim == "pearson": 
        itemsim = pickle.load(open( "save_itemsim_pearson.p", "rb" ))  
        
    elif sim == "distance":
        itemsim = pickle.load(open( "save_itemsim_distance.p", "rb" ))
    else: 
        print("User-based similarity matrix not computed, run SIMU command first")
        return None
    return itemsim

def get_Hybrid_Recommendations(prefs, cosim_matrix, itemsim, user, movies, movie_title_to_id, weight, item_thresh):
    '''
    Generates hybrid recommendations for a specified user

    Parameters:
    -- prefs: dictionary containing user-item matrix
    -- cosim_matrix: : pre-computed cosine similarity matrix from TF-IDF command
    -- itemsim: item-item similarity dictionary generated frm SIM command
    -- user: string containing name of user requesting recommendation
    -- movie: rated movie
    -- movie_title_to_id: dictionary that maps movie title to movieid
    -- weight: weight to apply to the item-item similarity
    -- SIG_THRESHOLD: neighborhood similarity threshold
    
    Returns:
    -- predictions: A list of recommended items with 0 or more tuples,
        each tuple contains (predicted rating, item name).
        List is sorted, high to low, by predicted rating.
        An empty list is returned when no recommendations have been calc'd. 
    '''

    predictions = []
    items_to_rate = []

    matrix2 = copy.copy(cosim_matrix)
    movie2 = movie_to_ID(movies)

    # converts similarity dictionary to a matrix
    for i in itemsim:
        location1 = int(movie_title_to_id[i]) - 1
        for j in range(len(itemsim[i])):
            location2 = int(movie2[itemsim[i][j][1]]) - 1
            matrix2[location1][location2] = itemsim[i][j][0]
            if i == movie2[itemsim[i][j][1]]:
                matrix2[location1][location2] = 1
    
    for item, itemID in movie_title_to_id.items():
        if item not in prefs[user].keys():
            items_to_rate.append((item, int(itemID)-1))

    for item, itemID in items_to_rate:
        den = 0
        num = 0
        for ratedMov, rating in prefs[user].items():
            i = cosim_matrix[int(movie_title_to_id[ratedMov])-1][itemID]
            j = matrix2[int(movie_title_to_id[ratedMov])-1][itemID]
            j = j * float(weight)
            # if cosim is 0, use item-item sim multiplied by hybrid weight
            if i == 0 and j>float(item_thresh):
                num+= (j*rating)
                den+=j
            elif i>float(item_thresh):
                num+= (i*rating)
                den+=i
        if den!=0:
            predictions.append((num/den,item))
    predictions.sort(reverse=True)
    return predictions
    
def single_Hybrid_Recommendations(prefs, cosim_matrix, itemsim, user, movies, item, movie_title_to_id, weight, item_thresh):
    '''
    Generates hybrid recommendations for a specified user

    Parameters:
    -- prefs: dictionary containing user-item matrix
    -- cosim_matrix: : pre-computed cosine similarity matrix from TF-IDF command
    -- itemsim: item-item similarity dictionary generated frm SIM command
    -- user: string containing name of user requesting recommendation
    -- movies: movies in the dataset
    -- item: specific item to generate prediction
    -- movie_title_to_id: dictionary that maps movie title to movieid
    -- weight: weight to apply to the item-item similarity
    -- SIG_THRESHOLD: neighborhood similarity threshold
    
    Returns:
    -- predictions: A list of recommended items with 0 or more tuples,
        each tuple contains (predicted rating, item name).
        List is sorted, high to low, by predicted rating.
        An empty list is returned when no recommendations have been calc'd. 
    '''

    items_to_rate = []
    items_to_rate.append((item, (int(movie_title_to_id[item])-1)))

    predictions = []

    matrix2 = copy.copy(cosim_matrix)
    movie2 = movie_to_ID(movies)

    #converts similarity dictionary to a matrix
    # for i in itemsim:
    #     location1 = int(movie_title_to_id[i]) - 1
    #     for j in range(len(itemsim[i])):
    #         location2 = int(movie2[itemsim[i][j][1]]) - 1
    #         matrix2[location1][location2] = itemsim[i][j][0]
    #         if i == movie2[itemsim[i][j][1]]:
    #             matrix2[location1][location2] = 1
    
    for mov,movId in items_to_rate:
        den = 0
        num = 0
        for ratedMovie,rating in prefs[user].items():
            i = cosim_matrix[int(movie_title_to_id[ratedMovie])-1][movId]
            j = itemsim[int(movie_title_to_id[ratedMovie])-1][movId]
            j = j * float(weight)
            #if cosim is 0, use item-item sim multiplied by hybrid weight
            if i == 0 and j>float(item_thresh):
                num+= (j*rating)
                den+=j
            elif i > float(item_thresh):
                num+= (i*rating)
                den+=i
        if den!=0:
            predictions.append((num/den,item))

    predictions.sort(reverse=True)

    return predictions

def convert_itemsim(matrix, movie, cosim_matrix, movie_title_to_id):
    '''
    Converts given tuple dictionary of similarities into a 2-D matrix.
    
    Parameters:
    -- matrix: pre-computed item-item similarity dictionary with tuples
    -- movie: list of movies
    -- cosim_matrix: : pre-computed cosine similarity matrix from TF-IDF command
    -- movie_title_to_id: dictionary that maps movie title to movieid
    
    Returns:
    -- matrix2: matrix of the similarities
    '''
    #copies format of cosim matrix
    matrix2 = copy.copy(cosim_matrix)
    movie2 = movie_to_ID(movie)
    #places each similarity to its position matching cosim
    for i in matrix:
        location1 = int(movie_title_to_id[i]) - 1
        for j in range(len(matrix[i])):
            location2 = int(movie2[matrix[i][j][1]]) - 1
            matrix2[location1][location2] = matrix[i][j][0]
            if i == movie2[matrix[i][j][1]]:
                matrix2[location1][location2] = 1
    return matrix2

def loo_cv_sim_tfidf(prefs, sim_matrix, SIG_THRESHOLD, movie_to_id):
    mse_error_list, mae_error_list = [], []
    i = 0
    start_time = timeit.default_timer()
    temp = copy.deepcopy(prefs)
    for user in prefs.keys():
        i += 1
        # if i%50 == 0:
        #     print("%d / %d" % (i, len(prefs.keys())))
        for item in list(prefs[user].keys()):
            #actual = prefs[user][item]
            actual = temp[user].pop(item) # remove item 
            recs = single_TFIDF_rec(temp, sim_matrix, user, SIG_THRESHOLD, movie_to_id, item) # get recommendation
            temp[user][item] = actual # restore item]
            for rec in recs: 
                
                if rec[1] == item: 
                    prediction = rec[0]
                    mse_error = (prediction - actual) **2 
                    mae_error = np.abs(prediction - actual)
                    mse_error_list.append(mse_error)
                    mae_error_list.append(mae_error)
                    

    if len(mse_error_list) == 0:
        return 0, 0, 0, []
    mse = sum(mse_error_list)/len(mse_error_list)
    mae = sum(mae_error_list)/len(mae_error_list)
    rmse = math.sqrt(sum(mse_error_list)/len(mse_error_list))

    return mse, mae, rmse, mse_error_list

def loo_cv_sim_hybrid(prefs, cosim_matrix, itemsim, movies, movie_to_id, weight, item_thresh):
    mse_error_list, mae_error_list = [], []
    i = 0
    start_time = timeit.default_timer()
    temp = copy.deepcopy(prefs)
    itemsim = convert_itemsim(itemsim, movies, cosim_matrix, movie_to_id)
    for user in prefs.keys():
        i += 1
        # if i%50 == 0:
        #     print("%d / %d" % (i, len(prefs.keys())))
        for item in list(prefs[user].keys()):
            #actual = prefs[user][item]
            actual = temp[user].pop(item) # remove item 
            recs = single_Hybrid_Recommendations(temp, cosim_matrix, itemsim, user, movies, item, movie_to_id, weight, item_thresh) # get recommendation
            temp[user][item] = actual # restore item]

            for rec in recs: 
                
                if rec[1] == item: 
                    prediction = rec[0]
                    mse_error = (prediction - actual) **2 
                    mae_error = np.abs(prediction - actual)
                    mse_error_list.append(mse_error)
                    mae_error_list.append(mae_error)
                    

    if len(mse_error_list) == 0:
        return 0, 0, 0, []
    mse = sum(mse_error_list)/len(mse_error_list)
    mae = sum(mae_error_list)/len(mae_error_list)
    rmse = math.sqrt(sum(mse_error_list)/len(mse_error_list))

    return mse, mae, rmse, mse_error_list
    
def eval_mf(MF):

    mse_error_list, mae_error_list = [], []
    
    for u in range(MF.user_vecs.shape[0]):
        for i in range(MF.item_vecs.shape[0]):
            actual = MF.ratings[u][i]
            if actual != 0:
                pred = MF.predict(u, i)
                
                se = (pred-actual) **2
                error = abs(pred-actual)

                mse_error_list.append(se)
                mae_error_list.append(error)
    
    
    if len(mse_error_list) == 0:
        return 0, 0, 0, []
    mse = sum(mse_error_list)/len(mse_error_list)
    mae = sum(mae_error_list)/len(mae_error_list)
    rmse = math.sqrt(sum(mse_error_list)/len(mse_error_list))

    return mse, mae, rmse, mse_error_list
    
def loo_cv_sim(prefs, sim, algo, sim_matrix, weight, thresh):
    """
    Leave-One_Out Evaluation: evaluates recommender system ACCURACY
     
     Parameters:
         prefs dataset: critics, etc.
     metric: MSE, or MAE, or RMSE
     sim: distance, pearson, etc.
     algo: user-based recommender, item-based recommender, etc.
     sim_matrix: pre-computed similarity matrix
     
    Returns:
         error: MSE, MAE, RMSE totals for this set of conditions
        mse_error_list: SE list for set of conditions

    """

    mse_error_list, mae_error_list = [], []
    i = 0
    start_time = timeit.default_timer()
    temp = copy.deepcopy(prefs)
    for user in prefs.keys():
        i += 1
        # if i%50 == 0:
        #     print("%d / %d" % (i, len(prefs.keys())))
        for item in list(prefs[user].keys()):
            #actual = prefs[user][item]
            actual = temp[user].pop(item) # remove item 
            recs = algo(temp, user, sim_matrix, item, weight, thresh) # get recommendation
            temp[user][item] = actual # restore item]
            for rec in recs: 
                
                if rec[1] == item: 
                    prediction = rec[0]
                    mse_error = (prediction - actual) **2 
                    mae_error = np.abs(prediction - actual)
                    mse_error_list.append(mse_error)
                    mae_error_list.append(mae_error)
                    
                    # print("User: %s, Item: %s, Prediction: %.3f, Actual: %.3f, Error: %.3f " % (user, item, prediction, actual, error))
            
            # if item not in set([rec[1] for rec in recs]):
            #         # print("From loo_cv_sim(), No prediction calculated for item %s, user %s in pred_list: %s" % (item, user, recs))
            #         continue
        if len(mse_error_list) > 0:
            if i%100 == 0:
                print("Number of users processed: %d " % (i))
                cur_time = timeit.default_timer()
                print("%.2f secs for %d users, time per user: %.2f" % (cur_time - start_time, i, (cur_time - start_time)/i))
                # print("Squared Error List:", mse_error_list)
                mse = sum(mse_error_list)/len(mse_error_list) 
                mae = sum(mae_error_list)/len(mae_error_list)
                rmse = math.sqrt(sum(mse_error_list)/len(mse_error_list))
                print("MSE: %f, RMSE: %f, MAE: %f" %(mse, rmse, mae))

    if len(mse_error_list) == 0:
        return 0, 0, 0, []
    mse = sum(mse_error_list)/len(mse_error_list)
    mae = sum(mae_error_list)/len(mae_error_list)
    rmse = math.sqrt(sum(mse_error_list)/len(mse_error_list))
    
    
    return mse, mae, rmse, mse_error_list

def get_mf_recommendations(MF, movies, user):

    predictions = []
    for i in range(1,MF.item_vecs.shape[0]):
        rating = MF.predict(int(user), i)
        item = movies[str(i)]
        predictions.append((rating, item))
    
    predictions.sort(reverse=True)
    return predictions


def main():
    ''' User interface for Python console '''

    # Load critics dict from file
    path = os.getcwd() # this gets the current working directory
                       # you can customize path for your own computer here
    print('\npath: %s' % path) # debug
    done = False
    prefs = {}
    
    while not done: 
        print()
        # Start a simple dialog
        file_io = input('R(ead) critics data from file?, \n'
                        'RML(ead ml100K data)?, \n'
                        'PD-R(ead) critics data from file?, \n'
                        'PD-RML100(ead) ml100K data from file?, \n'
                        'V(alidate) the dictionary?, \n'
                        'S(tats) print?, \n'
                        'SIM(ilarity matrix) calc for Item-based recommender?, \n'
                        'SIMU(ser-User matrix) calc for User-based recommender?, \n'
                        'T(est/train datasets?, \n'
                        'MF-ALS(atrix factorization- Alternating Least Squares)? \n'
                        'MF-SGD(atrix factorization- Stochastic Gradient Descent)? \n'
                        'TFIDF(and cosine sim Setup)?, \n'
                        'LCVSIM(eave one out cross-validation)?, \n'
                        'H(ybrid reccommendation setup), \n' 
                        'NCF(Neual Collaborative Filtering), \n'
                        'RECS(ecommendations -- all algos)? \n==> ')
        
        if file_io == 'R' or file_io == 'r':
            print()
            file_dir = 'data/'
            datafile = 'critics_ratings.data'
            itemfile = 'critics_movies.item'
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            print('Number of users: %d\nList of users:' % len(prefs), 
                  list(prefs.keys()))      
            data_ready = True
            
        elif file_io == 'PD-R' or file_io == 'pd-r':
            data_folder = '/data/' # for critics
            #print('\npath: %s\n' % path_name + data_folder) # debug: print path info
            names = ['user_id', 'item_id', 'rating', 'timestamp'] # column headings
            
            #Create pandas dataframe
            df = pd.read_csv(path + data_folder + 'critics_ratings_userIDs.data', sep='\t', names=names) # for critics
            ratings = file_info(df)
            
            # set test/train in case they were set by a previous file I/O command
            test_train_done = False
            data_ready = True
            print()
            print('Test and Train arrays are empty!')
            print()
       
        elif file_io == 'PD-RML100' or file_io == 'pd-rml100':
            
            # Load user-item matrix from file
            ## Read in data: ml-100k
            data_folder = '/data/ml-100k/' # for ml-100k                   
            #print('\npath: %s\n' % path_name + data_folder) # debug: print path info
            names = ['user_id', 'item_id', 'rating', 'timestamp'] # column headings
            file_dir = 'data/ml-100k/' # path from current directory
            datafile = 'u.data'  # ratings file
            itemfile = 'u.item'  # movie titles file    
            genrefile = 'u.genre' # movie genre file        
            #Create pandas dataframe
            df = pd.read_csv(path + data_folder + 'u.data', sep='\t', names=names) # for ml-100k
            ratings = file_info(df)
            movies, genres, features = from_file_to_2D(path, file_dir+genrefile, file_dir+itemfile)
            test_train_done = False
            data_ready = True
            print()
            print('Test and Train arrays are empty!')
            print()

        elif file_io == 'RML' or file_io == 'rml':
            print()
            file_dir = 'data/ml-100k/' # path from current directory
            datafile = 'u.data'  # ratings file
            itemfile = 'u.item'  # movie titles file    
            genrefile = 'u.genre' # movie genre file        
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            print('Number of users: %d\nList of users [0:10]:' 
                      % len(prefs), list(prefs.keys())[0:10] )  
            movies, genres, features = from_file_to_2D(path, file_dir+genrefile, file_dir+itemfile)
            print('Number of users: %d\nList of users [0:10]:' 
                  % len(prefs), list(prefs.keys())[0:10] ) 
            print ('Number of distinct genres: %d, number of feature profiles: %d' 
                   % (len(genres), len(features)))
            print('genres')
            print(genres)
            print('features')
            print(features)
            ready = False
            data_ready = True
        
        elif file_io == 'D' or file_io == 'd':
            print()
            if len(prefs) > 0:            
                # print('Examples:')
                # print ('Distance sim Lisa & Gene:', sim_distance(prefs, 'Lisa', 'Gene')) # 0.29429805508554946
                # num=1
                # den=(1+ sqrt( (2.5-3.0)**2 + (3.5-3.5)**2 + (3.0-1.5)**2 + (3.5-5.0)**2 + (3.0-3.0)**2 + (2.5-3.5)**2))
                # print('Distance sim Lisa & Gene (check):', num/den)    
                # print ('Distance sim Lisa & Michael:', sim_distance(prefs, 'Lisa', 'Michael')) # 0.4721359549995794
                # print()
                print('User-User distance similarities:')
                user_similarities = calc_user_dist_similarities(prefs)
                print()
                min_similarity = min(user_similarities, key=user_similarities.get)
                max_similarity = max(user_similarities, key=user_similarities.get)
                print("Least Similar Users are %s & %s, with RS distance of %.3f" % (min_similarity[0], min_similarity[1], 
                    user_similarities[min_similarity[0], min_similarity[1]]))
                print("Most Simiar Users: %s & %s, with RS distance of %.3f" % (max_similarity[0], max_similarity[1],
                    user_similarities[max_similarity[0], max_similarity[1]]))
                
            else:
                print ('Empty dictionary, R(ead) in some data!')    
        
        elif file_io == 'V' or file_io == 'v':      
            print()
            if len(prefs) > 0:
                # Validate the dictionary contents ..
                print ('Validating "%s" dictionary from file' % datafile)
                print ("critics['Lisa']['Lady in the Water'] =", 
                       prefs['Lisa']['Lady in the Water']) # ==> 2.5
                print ("critics['Toby']:", prefs['Toby']) 
                # ==> {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0, 
                #      'Superman Returns': 4.0}
            else:
                print ('Empty dictionary, R(ead) in some data!')              
        
        elif file_io == 'S' or file_io == 's':
            print()
            filename = 'critics_ratings.data'
            if len(prefs) > 0:
                data_stats(prefs, filename)
                popular_items(prefs, filename)
            else: # Make sure there is data  to process ..
                print ('Empty dictionary, R(ead) in some data!')   
        
        elif file_io == 'PC' or file_io == 'pc':
            print()
            if len(prefs) > 0:             
                print ('Example:')
                print ('Pearson sim Lisa & Gene:', sim_pearson(prefs, 'Lisa', 'Gene')) # 0.39605901719066977
                print()
                
                print('Pearson for all users:')
                # Calc Pearson for all users
                
                print('User-User distance similarities:')
                user_similarities = calc_user_pc_similarities(prefs)
                print()
                min_similarity = min(user_similarities, key=user_similarities.get)
                max_similarity = max(user_similarities, key=user_similarities.get)

                print("Least Similar Users are %s & %s, with pearson correlation of %.3f" % (min_similarity[0], min_similarity[1], 
                    user_similarities[min_similarity[0], min_similarity[1]]))
                print("Most Simiar Users: %s & %s, with pearson correlation  of %.3f" % (max_similarity[0], max_similarity[1],
                    user_similarities[max_similarity[0], max_similarity[1]]))
                
                print()
                
            else:
                print ('Empty dictionary, R(ead) in some data!')    
        
        elif file_io == 'U' or file_io == 'u':
            print()
            if len(prefs) > 0:             
                print ('Example:')
                user_name = 'Toby'
                print ('User-based CF recs for %s, sim_pearson: ' % (user_name), 
                       getRecommendations(prefs, user_name)) 
                        # [(3.3477895267131017, 'The Night Listener'), 
                        #  (2.8325499182641614, 'Lady in the Water'), 
                        #  (2.530980703765565, 'Just My Luck')]
                print ('User-based CF recs for %s, sim_distance: ' % (user_name),
                       getRecommendations(prefs, user_name, similarity=sim_distance)) 
                        # [(3.457128694491423, 'The Night Listener'), 
                        #  (2.778584003814924, 'Lady in the Water'), 
                        #  (2.422482042361917, 'Just My Luck')]
                print()
                
                print('User-based CF recommendations for all users:')
                # Calc User-based CF recommendations for all users
        
                ## add some code here to calc User-based CF recommendations 
                ## write a new function to do this ..
                print("Using sim_pearson:")
                get_all_UU_recs(prefs)
                print()
                get_all_UU_recs(prefs, sim_distance)
                
                ##    '''
                
                print()
                
            else:
                print ('Empty dictionary, R(ead) in some data!')     
       
        elif file_io == 'LCV' or file_io == 'lcv':
            print()
            if len(prefs) > 0:             
                print ('LOO_CV Evaluation: ')            
                ## add some code here to calc LOOCV 
                ## write a new function to do this ..
                error, error_list = loo_cv(prefs, 'MSE', sim_pearson, getRecommendations)
                print()
                error, error_list = loo_cv(prefs, 'MSE', sim_distance, getRecommendations)
            else:
                print ('Empty dictionary, R(ead) in some data!')            
        
        elif file_io == 'I' or file_io == 'i':
            print()
            if len(prefs) > 0 and len(itemsim) > 0:                
                print ('Example:')
                user_name = 'Toby'
    
                print ('Item-based CF recs for %s, %s: ' % (user_name, sim_method), 
                       getRecommendedItems(prefs, itemsim, user_name)) 
                
                print()
                
                print('Item-based CF recommendations for all users:')
                
                get_all_II_recs(prefs, itemsim, sim_method) # num_users=10, and top_N=5 by default  '''
                
                print()
                
            else:
                if len(prefs) == 0:
                    print ('Empty dictionary, R(ead) in some data!')
                else:
                    print ('Empty similarity matrix, use Sim(ilarity) to create a sim matrix!')    
        
        elif file_io == 'Sim' or file_io == 'sim' or file_io == 'SIM':
            print()
            if len(prefs) > 0: 
                ready = False # sub command in progress
                sub_cmd = input('RD(ead) distance or RP(ead) pearson or WD(rite) distance or WP(rite) pearson? ')
                try:
                    if sub_cmd == 'RD' or sub_cmd == 'rd':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(open( "save_itemsim_distance.p", "rb" ))
                        sim_method = 'sim_distance'
                        recType = 'item'
                        recAlgo = "item-based-distance"
                        weight = IBD_SIG_WEIGHT
                        thresh = IBD_SIG_THRESHOLD
                    elif sub_cmd == 'RP' or sub_cmd == 'rp':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(open( "save_itemsim_pearson.p", "rb" ))  
                        sim_method = 'sim_pearson'
                        recType = 'item'
                        recAlgo = "item-based-pearson"
                        weight = IBP_SIG_WEIGHT
                        thresh = IBP_SIG_THRESHOLD
                    elif sub_cmd == 'WD' or sub_cmd == 'wd':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        weight = IBD_SIG_WEIGHT
                        # n = len(prefs)
                        itemsim = calculateSimilarItems(prefs, weight, similarity=sim_distance)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(itemsim, open( "save_itemsim_distance.p", "wb" ))
                        sim_method = "sim_distance"
                        recType = 'item'
                        recAlgo = "item-based-distance"
                        thresh = IBD_SIG_THRESHOLD
                    elif sub_cmd == 'WP' or sub_cmd == 'wp':
                        weight = IBP_SIG_WEIGHT
                        # n = len(prefs)
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(prefs, weight, similarity=sim_pearson)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(itemsim, open( "save_itemsim_pearson.p", "wb" )) 
                        sim_method = "sim_pearson"
                        recType = 'item'
                        recAlgo = "item-based-pearson"
                        thresh = IBP_SIG_THRESHOLD
                    else:
                        print("Sim sub-command %s is invalid, try again" % sub_cmd)
                        continue
                    
                    ready = True # sub command completed successfully
                    
                except Exception as ex:
                    print ('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!'
                           ' Enter Sim(ilarity matrix) again and choose a Write command')
                    print()
                

                if len(itemsim) > 0 and ready == True: 
                    # Only want to print if sub command completed successfully
                    print ('Similarity matrix based on %s, len = %d' 
                           % (sim_method, len(itemsim)))
                    print()
                    # for item in itemsim:
                    #     print(item, itemsim[item])
                print()
    
               
                
            else:
                print ('Empty dictionary, R(ead) in some data!') 
        
        elif file_io == 'Simu' or file_io == 'simu':
            print()
            if len(prefs) > 0: 
                ready = False # sub command in progress
                sub_cmd = input('RD(ead) distance or RP(ead) pearson or WD(rite) distance or WP(rite) pearson? ')
                try:
                    if sub_cmd == 'RD' or sub_cmd == 'rd':
                        # Load the dictionary back from the pickle file.
                        usersim = pickle.load(open( "save_usersim_distance.p", "rb" ))
                        sim_method = 'sim_distance'
                        recAlgo = "user-based-distance"
                        recType = 'user'
                        weight = UBD_SIG_WEIGHT
                        thresh = UBD_SIG_THRESHOLD

                    elif sub_cmd == 'RP' or sub_cmd == 'rp':
                        # Load the dictionary back from the pickle file.
                        usersim = pickle.load(open( "save_usersim_pearson.p", "rb" ))  
                        sim_method = 'sim_pearson'
                        recAlgo = "user-based-pearson"
                        recType = 'user'
                        weight = UBP_SIG_WEIGHT
                        thresh = UBP_SIG_THRESHOLD

                    elif sub_cmd == 'WD' or sub_cmd == 'wd':
                        # transpose the U-I matrix and calc user-user similarities matrix
                        weight = UBD_SIG_WEIGHT
                        usersim = calculateSimilarUsers(prefs, weight, n=100,similarity=sim_distance)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(usersim, open( "save_usersim_distance.p", "wb" ))
                        recAlgo = "user-based-distance"
                        sim_method = "sim_distance"
                        weight = UBD_SIG_WEIGHT
                        thresh = UBD_SIG_THRESHOLD
                    elif sub_cmd == 'WP' or sub_cmd == 'wp':
                        # transpose the U-I matrix and calc user-user similarities matrix
                        weight = 1
                        usersim = calculateSimilarUsers(prefs,weight, n=100,similarity=sim_pearson)
                        # Dump/save dictionary to a pickle file
                        pickle.dump(usersim, open( "save_usersim_pearson.p", "wb" )) 
                        recAlgo = "user-based-pearson"
                        sim_method = "sim_pearson"
                        weight = UBP_SIG_WEIGHT
                        thresh = UBP_SIG_THRESHOLD 
                    else:
                        print("Simu sub-command %s is invalid, try again" % sub_cmd)
                        continue
                    
                    ready = True # sub command completed successfully
                    
                except Exception as ex:
                    print ('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!'
                           ' Enter Simu(ser-User matrix) again and choose a Write command')
                    print()
                if len(usersim) > 0 and ready == True: 
                    # Only want to print if sub command completed successfully
                    print ('Similarity matrix based on %s, len = %d' 
                           % (sim_method, len(usersim)))
                    print()
                    # for user in usersim:
                    #     print(user, usersim[user])
                print()
                
            else:
                print ('Empty dictionary, R(ead) in some data!') 

        
        elif file_io == 'LCVSIM' or file_io == 'lcvsim':
            print()
            if ready: 
                if recAlgo == "user-based-pearson":
                    sim_matrix = get_uu_cf_matrix("pearson")
                    sim = sim_pearson
                    algo = ext_getRecommendationsSim
                    mse, mae, rmse, error_list = loo_cv_sim(prefs, sim, algo, sim_matrix, weight, thresh)
                    #num_ratings = len(prefs)
                elif recAlgo == "user-based-distance":
                    sim_matrix = get_uu_cf_matrix("distance")
                    sim = sim_distance
                    algo = ext_getRecommendationsSim
                    mse, mae, rmse, mse_error_list  = loo_cv_sim(prefs, sim, algo, sim_matrix, weight, thresh)
                elif recAlgo == "item-based-pearson":
                    sim_matrix = get_ii_cf_matrix("pearson")
                    sim = sim_pearson
                    algo = ext_getRecommendedItems
                    mse, mae, rmse, error_list = loo_cv_sim(prefs, sim, algo, sim_matrix, weight, thresh)
                elif recAlgo == 'item-based-distance':
                    sim_matrix = get_ii_cf_matrix("distance")
                    sim = sim_distance
                    algo = ext_getRecommendedItems
                    mse, mae, rmse, error_list = loo_cv_sim(prefs, sim, algo, sim_matrix, weight, thresh)
                elif recAlgo == 'tfidf':
                    sim_matrix = cosim_matrix
                    threshs = [0, 0.15, 0.3, 0.45]
                    results = pd.DataFrame(columns=['thresh', 'MSE', 'MAE', 'RMSE', 'Coverage'])
                    
                    for t in threshs: 
                        data={}
                        mse, mae, rmse, error_list = loo_cv_sim_tfidf(prefs, sim_matrix, t, movie_to_ID(movies))
                        coverage = len(error_list)
                        data['thresh'], data['MSE'], data['MAE'], data['RMSE'], data['Coverage'] = t, mse, mae, rmse, coverage
                        results = results.append(data, ignore_index=True)
                    results.to_csv('tfidf_exp.csv')
                elif recAlgo == 'hybrid':

                    weights = [1.0, 0.25, 0.5, 0.75]
                    results = pd.DataFrame(columns=['weight', 'MSE', 'MAE', 'RMSE', 'Coverage'])
                    for w in weights:
                        data = {}
                        mse, mae, rmse, error_list = loo_cv_sim_hybrid(prefs, cosim_matrix, itemsim, movies, movie_to_ID(movies), w, thresh)
                        coverage = len(error_list)
                        data['thresh'], data['MSE'], data['MAE'], data['RMSE'], data['Coverage'] = t, mse, mae, rmse, coverage
                        results = results.append(data, ignore_index=True)
                    results.to_csv('hybrid_pearson_weight_exp.csv')

                elif recAlgo == 'MF_ALS':
                    mse, mae, rmse, error_list = eval_mf(MF_ALS)
                elif recAlgo == 'MF_SGD':
                    mse, mae, rmse, error_list = eval_mf(MF_SGD)
                elif recAlgo == 'ncf':
                    mse, mae, rmse, error_list = NCF.eval_recs()
                print('%s-based LOO_CV_SIM Evaluation:' % recAlgo)
                
                coverage = len(error_list)
                print('%s for ML-100K: %.5f, len(SE list): %d ' % ("MSE", mse, len(error_list)) )
                print('%s for ML-100K: %.5f, len(SE list): %d ' % ("MAE", mae, len(error_list)) )
                print('%s for ML-100K: %.5f, len(SE list): %d ' % ("RMSE", rmse, len(error_list)) )
                # print('%s for ML-100K: %.5f, len(SE list): %d ' % ("len(SE list)", coverage, len(error_list)) )
                np.savetxt(f'{recAlgo}_mse_error_list.csv', np.asarray(error_list), delimiter=',')
                print()
                    
            else:
                print('Run Sim(ilarity matrix) command to create/load Sim matrix!')
            
        elif file_io == 'T' or file_io == 't':
            if len(ratings) > 0:
                answer = input('Generate both test and train data? Y or y, N or n: ')
                if answer == 'N' or answer == 'n':
                    TRAIN_ONLY = True
                else:
                    TRAIN_ONLY = False
                
                #print('TRAIN_ONLY  in EVAL =', TRAIN_ONLY) ## debug
                train, test = mf_train_test_split(ratings, TRAIN_ONLY) ## this should 
                ##     be only place where TRAIN_ONLY is needed!! 
                ##     Check for len(test)==0 elsewhere
                
                test_train_info(test, train) ## print test/train info
        
                ## How is MSE calculated for train?? self.ratings is the train
                ##    data when ExplicitMF is instantiated for both als and sgd.
                ##    So, MSE calc is between predictions based on train data 
                ##    against the actuals for train data
                ## How is MSE calculated for test?? It's comparing the predictions
                ##    based on train data against the actuals for test data
                
                test_train_done = True
                print()
                print('Test and Train arrays are ready!')
                print()
            else:
                print ('Empty U-I matrix, read in some data!')
                print()            
    
        elif file_io == 'MF-ALS' or file_io == 'mf-als':
            
            if len(ratings) > 0:
                if test_train_done:
                    
                    ## als processing
                    
                    print()
                    ## sample instantiations ..
                    if len(ratings) < 10: ## for critics
                        print('Sample for critics .. ')
                        iter_array = [1, 2, 5, 10, 20]
                        MF_ALS = ExplicitMF(train, learning='als', n_factors=ALS_FACTORS, user_fact_reg=ALS_REG, item_fact_reg=ALS_REG, max_iters=max(iter_array), verbose=True)
                        print('[2,1,20]')
                    
                    elif len(ratings) < 1000: ## for ml-100k
                        print('Sample for ml-100k .. ')
                        iter_array = [1, 2, 5 , 10, 20, 50] #, 100] #, 200]
                        MF_ALS = ExplicitMF(train, learning='als', n_factors=ALS_FACTORS, user_fact_reg=ALS_REG, item_fact_reg=ALS_REG, max_iters=max(iter_array), verbose=True)
                        print('[20,0.01,50]')
                    
                    elif len(ratings) < 10000: ## for ml-1m
                        print('Sample for ml-1M .. ')
                        iter_array = [1, 2, 5, 10] 
                        MF_ALS = ExplicitMF(train, learning='als', n_factors=ALS_FACTORS, user_fact_reg=ALS_REG, item_fact_reg=ALS_REG, max_iters=max(iter_array), verbose=True)
                        print('[20,0.1,10]')
                        
                    #parms = input('Y or y to use these parameters or Enter to modify: ')# [2,0.01,10,False]
                    # if parms == 'Y' or parms == 'y':
                    #     pass
                    # else:
                        # # parms = eval(input('Enter new parameters as a list: [n_factors, reg, iters]: '))
                        
                        # # instantiate with this set of parms
                        # MF_ALS = ExplicitMF(train,learning='als', 
                        #                     n_factors=parms[0], 
                        #                     user_fact_reg=parms[1], 
                        #                     item_fact_reg=parms[1])
                       
                        # # set up the iter_array for this run to pass on
                        # orig_iter_array = [1, 2, 5, 10, 20, 50, 100, 200]
                        # i_max = parms[2]
                        # index = orig_iter_array.index(i_max)
                        # iter_array = []
                        # for i in range(0, index+1):
                        #     iter_array.append(orig_iter_array[i])
                            
                    # run the algo and plot results
                    MF_ALS.calculate_learning_curve(iter_array, test) 
                    # plot_learning_curve(iter_array, MF_ALS )
                    recAlgo = "MF_ALS"
                    ready = True
                else:
                    print ('Empty test/train arrays, run the T command!')
                    print()                    
            else:
                print ('Empty U-I matrix, read in some data!')
                print()
                    
        elif file_io == 'MF-SGD' or file_io == 'mf-sgd':
            
            if len(ratings) > 0:
                
                if test_train_done:
                
                    ## sgd processing
                     
                    ## sample instantiations ..
                    if len(ratings) < 10: ## for critics
                        # Use these parameters for small matrices
                        print('Sample for critics .. ')
                        iter_array = [1, 2, 5, 10, 20]                     
                        MF_SGD = ExplicitMF(train, 
                                            n_factors=SGD_FACTORS, 
                                            learning='sgd', 
                                            sgd_alpha=SGD_LEARNING_RATE,
                                            sgd_beta=SGD_REG, 
                                            max_iters=max(iter_array), 
                                            sgd_random=False)
                        print('[2, 0.075, 0.01, 20]')
                        print()

                    elif len(ratings) < 1000:
                       # Use these parameters for ml-100k
                        print('Sample for ml-100k .. ')
                        iter_array = [1, 2, 5, 10, 20]                     
                        MF_SGD = ExplicitMF(train, 
                                             n_factors=SGD_FACTORS, 
                                            learning='sgd', 
                                            sgd_alpha=SGD_LEARNING_RATE,
                                            sgd_beta=SGD_REG, 
                                            max_iters=max(iter_array), 
                                            sgd_random=False, verbose=True)
                        print('[2, 0.02, 0.2, 20]')
                        
                        '''
                        [2, 0.02, 0.2, 20]
                        Iteration: 20
                        Train mse: 0.8385557879257434
                        Elapsed train/test time 49.83 secs
                        
                        
                        [2, 0.02, 0.2, 20]
                        Iteration: 20
                        Train mse: 0.8297642823863384
                        Test mse: 0.9327376374604811
                        Elapsed train/test time 43.19 secs
                        
                        
                        [2, 0.02, 0.2, 100]
                        Iteration: 100
                        Train mse: 0.790743571610336
                        Test mse: 0.9145944592112709
                        Elapsed train/test time 169.78 secs
                        
                        '''
                         
                    elif len(ratings) < 10000:
                       # Use these parameters for ml-1m
                        print('Sample for ml-1m .. ')
                        iter_array = [1, 2, 5, 10, 20] #, 20, 50, 100]                     
                        MF_SGD = ExplicitMF(train, 
                                            n_factors=SGD_FACTORS, 
                                            learning='sgd', 
                                            sgd_alpha=SGD_LEARNING_RATE,
                                            sgd_beta=SGD_REG, 
                                            max_iters=max(iter_array), 
                                            sgd_random=False, verbose=True)
                        print('[20, 0.1, 0.1, 10]') # 100]')
                        
                        '''
                        
                        [20, 0.1, 0.1, 100]
                        Iteration: 100
                        Train mse: 0.7670492245886339
                        Test mse: 0.8881805974749858
                        Elapsed train/test time 1670.80 secs
                        
                        
                        [200, 0.1, 0.1, 100]
                        Iteration: 100
                        Train mse: 0.7588100351334094
                        Test mse: 0.8834154624525616
                        Elapsed train/test time 1919.23 secs
                        '''
                     
                    # parms = input('Y or y to use these parameters or Enter to modify: ')# [2,0.01,10,False]
                    # if parms == 'Y' or parms == 'y':
                    #     pass
                    # else:
                    #     parms = eval(input('Enter new parameters as a list: [n_factors K, learning_rate alpha, reg beta, max_iters: ')) #', random]: '))
                    #     MF_SGD = ExplicitMF(train, 
                    #                         n_factors=SGD_FACTORS, 
                    #                         learning='sgd', 
                    #                         sgd_alpha=SGD_LEARNING_RATE,
                    #                         sgd_beta=SGD_REG, 
                    #                         max_iters=max(iter_array), 
                    #                         sgd_random=False, verbose=True)

                    #     orig_iter_array = [1, 2, 5, 10, 20, 50, 100, 200]
                    #     i_max = parms[3]
                    #     index = orig_iter_array.index(i_max)
                    #     iter_array = []
                    #     for i in range(0, index+1):
                    #         iter_array.append(orig_iter_array[i])
                         
                    MF_SGD.calculate_learning_curve(iter_array, test) # start the training
                    # plot_learning_curve(iter_array, MF_SGD)    
                    recAlgo = "MF_SGD"
                    ready = True
                else:
                    print ('Empty test/train arrays, run the T command!')
                    print() 

        elif file_io == 'ncf' or file_io == 'NCF':
            cwd = os.getcwd() 
            path = cwd + "/data/ml-100k/u.data"
            dataset = ncf_read_data(path)
            file_dir = 'data/ml-100k/' # path from current directory
            datafile = 'u.data'  # ratings file
            itemfile = 'u.item'  # movie titles file    
            genrefile = 'u.genre' # movie genre file     
            movies, genres, features = from_file_to_2D(cwd, file_dir+genrefile, file_dir+itemfile)
            NCF = NeuMF(dataset, 
                test_size=0.2,
                n_factors=5, 
                lr=0.01, 
                n_layers=3, 
                n_nodes_per_layer=[128, 64, 32, 16, 8, 4, 2],
                n_epochs=25, 
                batch_size=256, 
                model_num=1, 
                dropout_prob=0.2)
            
            NCF.train() 
            NCF.plot_learning_curve()
            recAlgo = 'ncf'
            ready = True

        elif file_io == 'TFIDF' or file_io == 'tfidf':
            R = to_array(prefs)
            feature_str = to_string(features)                 
            feature_docs = to_docs(feature_str, genres)
            
            print(R[:3][:5])
            print()
            print('features')
            print(features[0:5])
            print()
            print('feature docs')
            print(feature_docs[0:5]) 
            cosim_matrix = cosine_sim(feature_docs)
            print()
            print('cosine sim matrix')
            print (type(cosim_matrix), len(cosim_matrix))
            print()
            similarity_histogram(cosim_matrix)
            # print(single_Hybrid_Recommendations(prefs, cosim_matrix, itemsim, '340', movies, 'Once Upon a Time in the West (1969)', movie_to_ID(movies), 1, SIG_THRESHOLD))               
            ready = True
            recAlgo = 'tfidf'

        elif file_io == 'H' or file_io == 'h':

            
            # tfidf cosim matrix set up
            # R = to_array(prefs)
            # feature_str = to_string(features)                 
            # feature_docs = to_docs(feature_str, genres)
            # cosim_matrix = cosine_sim(feature_docs)
            # # item-based set up 
            # # transpose the U-I matrix and calc item-item similarities matrix
            # weight = 50
            # thresh = 0.0
            # itemsim = calculateSimilarItems(prefs, weight,similarity=sim_distance)                     
            # Dump/save dictionary to a pickle file
            # pickle.dump(itemsim, open( "save_itemsim_distance.p", "wb" ))
            ready = True
            recAlgo = 'hybrid'
            
        elif file_io == 'RECS' or file_io == 'recs': 
            if not data_ready: 
                print("No data is loaded")
            else:
                # recAlgo = input('Which recommendation algorithm would you like to use?, \n' 
                #                 'Item-based distance (ibd), \n'
                #                 'Item-based pearson (ibp), \n'
                #                 'User-based distance (ubd), \n'
                #                 'User-based pearson (ubp), \n'
                #                 'MF-Alternating Least Square (mf-als), \n', 
                #                 'MF-Stochastic Gradient Decent(mf-sgd), \n', 
                #                 'Term-frequency Inverse Document Frequency (tfidf), \n', 
                #                 'Hybrid Recommendation with IB distance (h-ibd), \n', 
                #                 'Hybrid Recommendation with IB pearson (h-ibp), \n', 
                #                 'Neural Collaborative Filtering (ncf), \n')

                # if recAlgo == 'ibp':
                # elif recAlgo == 'ibd':
                # elif recAlgo == 'ubp':
                # elif recAlgo == 'ubd':
                # elif recAlgo == 'mf-als':
                # elif recAlgo == 'mf-sgd':
                # elif recAlgo == 'tfidf':
                # elif recAlgo == 'h-ibd':
                # elif recAlgo == 'h-ibp':
                # elif recAlgo == 'ncf':
                # else:
                user = input('Enter userid (for ml-100k) or return to quit: ')
                n = input('How many recommendations would you like (1 for single, n for top-n)? ')
                #I think we need another input here to specify the algorithm
                #recAlgo = input("Enter ALS or SGD orDeep Learning or TFIDF or Hybrid: "")
                n = int(n)
                if ready: 
                    if recAlgo == "item-based-pearson":
                        sim_matrix = get_ii_cf_matrix("pearson")
                        
                        thresh = 0.0  
                        recommendation = getRecommendedItems(prefs, user, sim_matrix, weight, thresh)[:n]
                    elif recAlgo == "item-based-distance":
                        
                        thresh = 0.0
                        sim_matrix = get_ii_cf_matrix("distance")
                        recommendation = getRecommendedItems(prefs, user, sim_matrix, weight, thresh)[:n]
                    elif recAlgo == "user-based-pearson":
                        
                        thresh = 0.3
                        sim_matrix = get_uu_cf_matrix("pearson")
                        recommendation = getRecommendationsSim(prefs, user, sim_matrix, weight, thresh)[:n]
                    elif recAlgo == "user-based-distance":
                        
                        thresh = 0.0
                        sim_matrix = get_uu_cf_matrix("distance")
                        recommendation = getRecommendationsSim(prefs, user, sim_matrix, weight, thresh)[:n]
                    elif recAlgo == "MF_ALS":
                        
                        predictions = get_mf_recommendations(MF_ALS, movies, user)
                        recommendation = predictions[:n]
                    elif recAlgo == "MF_SGD":
                        predictions = get_mf_recommendations(MF_SGD, movies, user)
                        
                        recommendation = predictions[:n]
                    elif recAlgo == "tfidf":
                        sim_matrix = cosim_matrix
                        recommendation = get_TFIDF_recommendations(prefs, sim_matrix, user, TFIDF_SIG_THRESHOLD, movie_to_ID(movies))[:n]
                    
    
                    elif recAlgo == 'hybrid':
                        recommendation = get_Hybrid_Recommendations(prefs, cosim_matrix, itemsim, user, movies, movie_to_ID(movies), HYBRID_WEIGHT, thresh)[:n]
                    
                    elif recAlgo == 'ncf':
                        recommendation = NCF.get_recommendations(user, movies)[:n]

                    print("Top %d Recommendations from %s are: " % (n,recAlgo))
                    for rec in recommendation:
                        print(rec)
                # print(recommendation)
            

                else: 
                    print("Similarity matrix not ready. ")
                    print()
        
        
        else:
            done = True
    
    print('\nGoodbye!')
        
if __name__ == '__main__':
    main()