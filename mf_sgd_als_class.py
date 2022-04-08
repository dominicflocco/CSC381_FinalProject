# -*- coding: utf-8 -*-
'''
Example Implementation of Matrix Factorization in Python using the 
Stochastic Gradient Descent algorithm (SGD) and the
Alternating Least Squares (ALS) algorithm

Collaborator/Author: Carlos E. Seminario (CES)

'''



import numpy as np
import pandas as pd
from os import getcwd
from numpy.linalg import solve ## needed for als
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from time import time
from copy import deepcopy


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

    def train(self, n_iter=10): 
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
                self.train(n_iter - iter_diff) # init training, run first iter
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
            return self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
        elif self.learning == 'sgd':
            prediction = self.global_bias + self.user_bias[u] + self.item_bias[i]
            prediction += self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
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

def train_test_split(ratings, TRAIN_ONLY):
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
    
    plt.show()

def main():

    path_name = getcwd() # this gets the current working directory
                         # you can customize path for your own computer here
    print('\npath: %s' % path_name) # debug
    print()
    ratings = []
    #m = 0; n = 0 # m is number of users, n is number of items
    done = False
        
    
    while not done:
        print()
        file_io = input('PD-R(ead) critics data from file?, \n'
                        'PD-RML100(ead) ml100K data from file?, \n'
                        'PD-RML1M(ead) ml1M data from file?, \n'
                        'T(est/train datasets?, \n'
                        'MF-ALS(atrix factorization- Alternating Least Squares)? \n'
                        'MF-SGD(atrix factorization- Stochastic Gradient Descent)? \n'
                        'GS-ALS(Grid Search- Alternating Least Squares)? \n'
                        'GS-SGD(Grid Search- Stochastic Gradient Descent)? \n'
                        '===>> ')
        
        if file_io == 'PD-R' or file_io == 'pd-r':
            
            # Load user-item matrix from file
            
            ## Read in data: critics
            data_folder = '/data/' # for critics
            #print('\npath: %s\n' % path_name + data_folder) # debug: print path info
            names = ['user_id', 'item_id', 'rating', 'timestamp'] # column headings
            
            #Create pandas dataframe
            df = pd.read_csv(path_name + data_folder + 'critics_ratings_userIDs.data', sep='\t', names=names) # for critics
            ratings = file_info(df)
            
            # set test/train in case they were set by a previous file I/O command
            test_train_done = False
            print()
            print('Test and Train arrays are empty!')
            print()
    
        elif file_io == 'PD-RML100' or file_io == 'pd-rml100':
            
            # Load user-item matrix from file
            ## Read in data: ml-100k
            data_folder = '/data/ml-100k/' # for ml-100k                   
            #print('\npath: %s\n' % path_name + data_folder) # debug: print path info
            names = ['user_id', 'item_id', 'rating', 'timestamp'] # column headings
    
            #Create pandas dataframe
            df = pd.read_csv(path_name + data_folder + 'u.data', sep='\t', names=names) # for ml-100k
            ratings = file_info(df)
            
            test_train_done = False
            print()
            print('Test and Train arrays are empty!')
            print()
    
        elif file_io == 'PD-RML1M' or file_io == 'pd-rml1m':

            
            # Load user-item matrix from file
            ## Read in data: critics, ml-1m
            data_folder = '/data/ml-1m/' # for ml-1m                      
            #print('\npath: %s\n' % path_name + data_folder) # debug: print path info
            names = ['user_id', 'item_id', 'rating', 'timestamp'] # column headings
    
            #Create pandas dataframe
            df = pd.read_csv(path_name + data_folder + 'ratings.dat', engine='python', header=None, sep="::", names=names) # for ml1m
            ratings = file_info(df)
            
            test_train_done = False
            print()
            print('Test and Train arrays are empty!')
            print()
            
        elif file_io == 'T' or file_io == 't':
            if len(ratings) > 0:
                answer = input('Generate both test and train data? Y or y, N or n: ')
                if answer == 'N' or answer == 'n':
                    TRAIN_ONLY = True
                else:
                    TRAIN_ONLY = False
                
                #print('TRAIN_ONLY  in EVAL =', TRAIN_ONLY) ## debug
                train, test = train_test_split(ratings, TRAIN_ONLY) ## this should 
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
                        MF_ALS = ExplicitMF(train, learning='als', n_factors=2, user_fact_reg=1, item_fact_reg=1, max_iters=max(iter_array), verbose=True)
                        print('[2,1,20]')
                    
                    elif len(ratings) < 1000: ## for ml-100k
                        print('Sample for ml-100k .. ')
                        iter_array = [1, 2, 5 , 10, 20, 50] #, 100] #, 200]
                        MF_ALS = ExplicitMF(train, learning='als', n_factors=20, user_fact_reg=.01, item_fact_reg=.01, max_iters=max(iter_array), verbose=True) 
                        print('[20,0.01,50]')
                    
                    elif len(ratings) < 10000: ## for ml-1m
                        print('Sample for ml-1M .. ')
                        iter_array = [1, 2, 5, 10] 
                        MF_ALS = ExplicitMF(train, learning='als', n_factors=20, user_fact_reg=.1, item_fact_reg=.1, max_iters=max(iter_array), verbose=True) 
                        print('[20,0.1,10]')
                        
                    parms = input('Y or y to use these parameters or Enter to modify: ')# [2,0.01,10,False]
                    if parms == 'Y' or parms == 'y':
                        pass
                    else:
                        parms = eval(input('Enter new parameters as a list: [n_factors, reg, iters]: '))
                        
                        # instantiate with this set of parms
                        MF_ALS = ExplicitMF(train,learning='als', 
                                            n_factors=parms[0], 
                                            user_fact_reg=parms[1], 
                                            item_fact_reg=parms[1])
                       
                        # set up the iter_array for this run to pass on
                        orig_iter_array = [1, 2, 5, 10, 20, 50, 100, 200]
                        i_max = parms[2]
                        index = orig_iter_array.index(i_max)
                        iter_array = []
                        for i in range(0, index+1):
                            iter_array.append(orig_iter_array[i])
                            
                    # run the algo and plot results
                    MF_ALS.calculate_learning_curve(iter_array, test) 
                    plot_learning_curve(iter_array, MF_ALS )
                    
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
                                            n_factors=2, 
                                            learning='sgd', 
                                            sgd_alpha=0.075,
                                            sgd_beta=0.01, 
                                            max_iters=max(iter_array), 
                                            sgd_random=False)
                        print('[2, 0.075, 0.01, 20]')
                        print()

                    elif len(ratings) < 1000:
                       # Use these parameters for ml-100k
                        print('Sample for ml-100k .. ')
                        iter_array = [1, 2, 5, 10, 20]                     
                        MF_SGD = ExplicitMF(train, 
                                            n_factors=2, 
                                            learning='sgd', 
                                            sgd_alpha=0.02,
                                            sgd_beta=0.2, 
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
                        iter_array = [1, 2, 5, 10] #, 20, 50, 100]                     
                        MF_SGD = ExplicitMF(train, 
                                            n_factors=20, 
                                            learning='sgd', 
                                            sgd_alpha=0.1,
                                            sgd_beta=0.1, 
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
                     
                    parms = input('Y or y to use these parameters or Enter to modify: ')# [2,0.01,10,False]
                    if parms == 'Y' or parms == 'y':
                        pass
                    else:
                        parms = eval(input('Enter new parameters as a list: [n_factors K, learning_rate alpha, reg beta, max_iters: ')) #', random]: '))
                        MF_SGD = ExplicitMF(train, n_factors=parms[0], 
                                            learning='sgd', 
                                            sgd_alpha=parms[1], 
                                            sgd_beta=parms[2], 
                                            max_iters=parms[3], 
                                            sgd_random=False, verbose=True)  

                        orig_iter_array = [1, 2, 5, 10, 20, 50, 100, 200]
                        i_max = parms[3]
                        index = orig_iter_array.index(i_max)
                        iter_array = []
                        for i in range(0, index+1):
                            iter_array.append(orig_iter_array[i])
                         
                    MF_SGD.calculate_learning_curve(iter_array, test) # start the training
                    plot_learning_curve(iter_array, MF_SGD)    
                     
                else:
                    print ('Empty test/train arrays, run the T command!')
                    print()   


        elif file_io == 'GS-SGD' or file_io == 'gs-sgd':
            num_factors = [2, 20, 200]
            alphas = [0.02, 0.002, 0.0002]
            betas = [0.2, 0.02, 0.002]
            iters = 20     
            col_headers = ["Test case ID", "Test MSE @ 1", 
                                            "Test MSE @ 2", 
                                            "Test MSE @ 5", 
                                            "Test MSE @ 10",
                                            "Test MSE @ 20", 
                                            "Train MSE @ 1", 
                                            "Train MSE @ 2",
                                            "Train MSE @ 5", 
                                            "Train MSE @ 10", 
                                            "Train MSE @ 20"]
            sgd_results = pd.DataFrame(columns=col_headers)
            
            for f in num_factors: 
                for a in alphas: 
                    for b in betas: 
                        data = {}
                        test_id = "SGD~" + str(f) + "~" + str(a) + "~" + str(b)
                        MF_SGD = ExplicitMF(train, n_factors=f, 
                                        learning='sgd', 
                                        sgd_alpha=a, 
                                        sgd_beta=b, 
                                        max_iters=iters, 
                                        sgd_random=False, verbose=True)  
                        orig_iter_array = [1, 2, 5, 10, 20, 50, 100, 200]
                        i_max = iters
                        index = orig_iter_array.index(i_max)
                        iter_array = []
                        for i in range(0, index+1):
                            iter_array.append(orig_iter_array[i])
                        test_mse, train_mse = MF_SGD.calculate_learning_curve(iter_array, test)
                        data["Test case ID"] = test_id 
                        for i in range(len(col_headers[1:])//2):
                            data[col_headers[i+1]] = test_mse[i]
                        for i in range(len(col_headers[1:])//2):
                            data[col_headers[i + (len(col_headers[1:])//2) + 1]] = train_mse[i]

                        sgd_results = sgd_results.append(data, ignore_index=True)
            sgd_results.to_csv("SGD-results-1M.csv")

                        

        elif file_io == 'GS-ALS' or file_io == 'gs-als':      
            num_factors = [2, 20, 100, 200]
            lambdas = [1.0, 0.1, 0.01, 0.001, 0.00001]
            iters = 20
            col_headers = ["Test case ID", "Test MSE @ 1", 
                                            "Test MSE @ 2", 
                                            "Test MSE @ 5", 
                                            "Test MSE @ 10",
                                            "Test MSE @ 20", 
                                            "Train MSE @ 1", 
                                            "Train MSE @ 2",
                                            "Train MSE @ 5", 
                                            "Train MSE @ 10", 
                                            "Train MSE @ 20"]
            als_results = pd.DataFrame(columns=col_headers)
            
            for f in num_factors: 
                for l in lambdas: 
                    
                    data = {}
                    test_id = "ALS~" + str(f) + "~" + str(l)
                    MF_ALS = ExplicitMF(train, learning='als', 
                                        n_factors=f, 
                                        user_fact_reg=l, 
                                        item_fact_reg=l, 
                                        max_iters=iters, 
                                        verbose=True) 
                    orig_iter_array = [1, 2, 5, 10, 20, 50, 100, 200]
                    i_max = iters
                    index = orig_iter_array.index(i_max)
                    iter_array = []
                    for i in range(0, index+1):
                        iter_array.append(orig_iter_array[i])
                    test_mse, train_mse = MF_ALS.calculate_learning_curve(iter_array, test)
                    data["Test case ID"] = test_id 
                    for i in range(len(col_headers[1:])//2):
                            data[col_headers[i+1]] = test_mse[i]
                    for i in range(len(col_headers[1:])//2):
                        data[col_headers[i + (len(col_headers[1:])//2) + 1]] = train_mse[i]

                    als_results = als_results.append(data, ignore_index=True)
            als_results.to_csv("ALS-results-1M.csv")
            
            ##
            ## Place for new Grid_search command ..
            ## for sgd: values of K, alpha learning rate, beta regularization, max iters
            ## for als: values of num_factors, user and item regularization, max iters

            
    
        else:
            done = True

    print()
    print('Goodbye!')    


if __name__ == '__main__':
    main()
    

'''


PD-R(ead) critics data from file?, 
PD-RML100(ead) ml100K data from file?, 
PD-RML1M(ead) ml1M data from file?, 
T(est/train datasets?, 
MF-ALS(atrix factorization- Alternating Least Squares)? 
MF-SGD(atrix factorization- Stochastic Gradient Descent)? 
===>> pd-r

   user_id  item_id  rating  timestamp
0        1        1     2.5          0
1        1        2     3.5          0
2        1        3     3.0          0
3        1        4     3.5          0
4        1        5     2.5          0
Summary Stats:

         user_id    item_id     rating  timestamp
count  35.000000  35.000000  35.000000       35.0
mean    3.742857   3.571429   3.228571        0.0
std     1.975480   1.719879   0.885694        0.0
min     1.000000   1.000000   1.000000        0.0
25%     2.000000   2.000000   3.000000        0.0
50%     4.000000   4.000000   3.000000        0.0
75%     5.000000   5.000000   3.750000        0.0
max     7.000000   6.000000   5.000000        0.0

2D_matrix shape (7, 6)

[[2.5 3.5 3.  3.5 2.5 3. ]
 [3.  3.5 1.5 5.  3.5 3. ]
 [2.5 3.  0.  3.5 0.  4. ]
 [0.  3.5 3.  4.  2.5 4.5]
 [3.  4.  2.  3.  2.  3. ]
 [3.  4.  0.  5.  3.5 3. ]
 [0.  4.5 0.  4.  1.  0. ]]

7 users
6 items
Sparsity: 16.67%

Test and Train arrays are empty!



PD-R(ead) critics data from file?, 
PD-RML100(ead) ml100K data from file?, 
PD-RML1M(ead) ml1M data from file?, 
T(est/train datasets?, 
MF-ALS(atrix factorization- Alternating Least Squares)? 
MF-SGD(atrix factorization- Stochastic Gradient Descent)? 
===>> t

Generate both test and train data? Y or y, N or n: y

Train info: 7 rows, 6 cols
Test info: 7 rows, 6 cols
test ratings count = 7
train ratings count = 28
test + train count 35
test/train percentages: 20.00 / 80.00


Test and Train arrays are ready!



PD-R(ead) critics data from file?, 
PD-RML100(ead) ml100K data from file?, 
PD-RML1M(ead) ml1M data from file?, 
T(est/train datasets?, 
MF-ALS(atrix factorization- Alternating Least Squares)? 
MF-SGD(atrix factorization- Stochastic Gradient Descent)? 
===>> mf-als

Sample for critics .. 

ALS instance parameters:
n_factors=2, user_reg=1.00000,  item_reg=1.00000, num_iters=20

[2,1,20]

Y or y to use these parameters or Enter to modify: y

Runtime parameters:
n_factors=2, user_reg=1.00000, item_reg=1.00000, max_iters=20, 
ratings matrix: 7 users X 6 items

Elapsed train/test time 0.00 secs
Iteration: 1
Train mse: 0.269846126407827
Test mse: 0.6441062507662181
Elapsed train/test time 0.00 secs
Iteration: 2
Train mse: 0.23700875017140496
Test mse: 0.5851825030301893
Elapsed train/test time 0.01 secs
Iteration: 5
Train mse: 0.22458271993694207
Test mse: 0.6244731234593274
Elapsed train/test time 0.01 secs
Iteration: 10
Train mse: 0.22127015439602365
Test mse: 0.7050267421088833
Elapsed train/test time 0.01 secs
Iteration: 20
Train mse: 0.2265831984835294
Test mse: 0.6858639110217174
Elapsed train/test time 0.02 secs


PD-R(ead) critics data from file?, 
PD-RML100(ead) ml100K data from file?, 
PD-RML1M(ead) ml1M data from file?, 
T(est/train datasets?, 
MF-ALS(atrix factorization- Alternating Least Squares)? 
MF-SGD(atrix factorization- Stochastic Gradient Descent)? 
===>> mf-sgd
Sample for critics .. 

SGD instance parameters:
num_factors K=2, learn_rate alpha=0.07500, reg beta=0.01000, num_iters=20, sgd_random=False

[2, 0.075, 0.01, 20]


Y or y to use these parameters or Enter to modify: y

Runtime parameters:
n_factors=2, learning_rate alpha=0.075, reg beta=0.01000, max_iters=20, sgd_random=False 
ratings matrix: 7 users X 6 items

Elapsed train/test time 0.00 secs
Iteration: 1
Train mse: 0.49141183526518234
Test mse: 0.4330730870212343
Elapsed train/test time 0.00 secs
Iteration: 2
Train mse: 0.3319580404614439
Test mse: 0.37335118767677455
Elapsed train/test time 0.00 secs
Iteration: 5
Train mse: 0.22226379648067776
Test mse: 0.3508749640874531
Elapsed train/test time 0.01 secs
Iteration: 10
Train mse: 0.16726990515196588
Test mse: 0.3863964299763883
Elapsed train/test time 0.01 secs
Iteration: 20
Train mse: 0.07761634315083399
Test mse: 0.6060941515470414
Elapsed train/test time 0.02 secs


PD-R(ead) critics data from file?, 
PD-RML100(ead) ml100K data from file?, 
PD-RML1M(ead) ml1M data from file?, 
T(est/train datasets?, 
MF-ALS(atrix factorization- Alternating Least Squares)? 
MF-SGD(atrix factorization- Stochastic Gradient Descent)? 
===>> pd-rml100

   user_id  item_id  rating  timestamp
0      196      242       3  881250949
1      186      302       3  891717742
2       22      377       1  878887116
3      244       51       2  880606923
4      166      346       1  886397596
Summary Stats:

            user_id        item_id         rating     timestamp
count  100000.00000  100000.000000  100000.000000  1.000000e+05
mean      462.48475     425.530130       3.529860  8.835289e+08
std       266.61442     330.798356       1.125674  5.343856e+06
min         1.00000       1.000000       1.000000  8.747247e+08
25%       254.00000     175.000000       3.000000  8.794487e+08
50%       447.00000     322.000000       4.000000  8.828269e+08
75%       682.00000     631.000000       4.000000  8.882600e+08
max       943.00000    1682.000000       5.000000  8.932866e+08

2D_matrix shape (943, 1682)

[[5. 3. 4. ... 0. 0. 0.]
 [4. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [5. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 5. 0. ... 0. 0. 0.]]

943 users
1682 items
Sparsity: 93.70%

Test and Train arrays are empty!



PD-R(ead) critics data from file?, 
PD-RML100(ead) ml100K data from file?, 
PD-RML1M(ead) ml1M data from file?, 
T(est/train datasets?, 
MF-ALS(atrix factorization- Alternating Least Squares)? 
MF-SGD(atrix factorization- Stochastic Gradient Descent)? 
===>> t

Generate both test and train data? Y or y, N or n: y

Train info: 943 rows, 1682 cols
Test info: 943 rows, 1682 cols
test ratings count = 15910
train ratings count = 84090
test + train count 100000
test/train percentages: 15.91 / 84.09


Test and Train arrays are ready!



PD-R(ead) critics data from file?, 
PD-RML100(ead) ml100K data from file?, 
PD-RML1M(ead) ml1M data from file?, 
T(est/train datasets?, 
MF-ALS(atrix factorization- Alternating Least Squares)? 
MF-SGD(atrix factorization- Stochastic Gradient Descent)? 
===>> mf-als

Sample for ml-100k .. 

ALS instance parameters:
n_factors=20, user_reg=0.01000,  item_reg=0.01000, num_iters=50

[20,0.01,50]

Y or y to use these parameters or Enter to modify: y

Runtime parameters:
n_factors=20, user_reg=0.01000, item_reg=0.01000, max_iters=50, 
ratings matrix: 943 users X 1682 items

Elapsed train/test time 0.00 secs
Iteration: 1
Train mse: 0.5351517180494575
Test mse: 5.789221388766593
Elapsed train/test time 2.05 secs
Iteration: 2
Train mse: 0.41174303954010805
Test mse: 4.183590320410549
Elapsed train/test time 4.06 secs
Iteration: 5
Train mse: 0.325136558786744
Test mse: 3.8933421088831945
Elapsed train/test time 6.38 secs
Iteration: 10
Train mse: 0.2885189141786948
Test mse: 4.197666246203809
Elapsed train/test time 8.82 secs
Iteration: 20
Train mse: 0.2664595694639813
Test mse: 4.534155250782738
Elapsed train/test time 11.86 secs
Iteration: 50
Train mse: 0.2482878270962831
Test mse: 5.459441151620064
Elapsed train/test time 17.26 secs


PD-R(ead) critics data from file?, 
PD-RML100(ead) ml100K data from file?, 
PD-RML1M(ead) ml1M data from file?, 
T(est/train datasets?, 
MF-ALS(atrix factorization- Alternating Least Squares)? 
MF-SGD(atrix factorization- Stochastic Gradient Descent)? 
===>> mf-sgd
Sample for ml-100k .. 

SGD instance parameters:
num_factors K=2, learn_rate alpha=0.02000, reg beta=0.20000, num_iters=20, sgd_random=False

[2, 0.02, 0.2, 20]

Y or y to use these parameters or Enter to modify: y

Runtime parameters:
n_factors=2, learning_rate alpha=0.020, reg beta=0.20000, max_iters=20, sgd_random=False 
ratings matrix: 943 users X 1682 items

Elapsed train/test time 0.00 secs
Iteration: 1
Train mse: 0.8958995371950466
Test mse: 0.9980559532186012
Elapsed train/test time 4.07 secs
Iteration: 2
Train mse: 0.8656081731397046
Test mse: 0.9653436563570722
Elapsed train/test time 8.16 secs
Iteration: 5
Train mse: 0.8476992420827341
Test mse: 0.9445033567644885
Elapsed train/test time 15.16 secs
Iteration: 10
Train mse: 0.8426602624263262
Test mse: 0.9394088394176977
Elapsed train/test time 25.22 secs
Iteration: 20
Train mse: 0.8297642823863384
Test mse: 0.9327376374604811
Elapsed train/test time 43.15 secs


PD-R(ead) critics data from file?, 
PD-RML100(ead) ml100K data from file?, 
PD-RML1M(ead) ml1M data from file?, 
T(est/train datasets?, 
MF-ALS(atrix factorization- Alternating Least Squares)? 
MF-SGD(atrix factorization- Stochastic Gradient Descent)? 
===>> pd-rml1m

   user_id  item_id  rating  timestamp
0        1     1193       5  978300760
1        1      661       3  978302109
2        1      914       3  978301968
3        1     3408       4  978300275
4        1     2355       5  978824291
Summary Stats:

            user_id       item_id        rating     timestamp
count  1.000209e+06  1.000209e+06  1.000209e+06  1.000209e+06
mean   3.024512e+03  1.865540e+03  3.581564e+00  9.722437e+08
std    1.728413e+03  1.096041e+03  1.117102e+00  1.215256e+07
min    1.000000e+00  1.000000e+00  1.000000e+00  9.567039e+08
25%    1.506000e+03  1.030000e+03  3.000000e+00  9.653026e+08
50%    3.070000e+03  1.835000e+03  4.000000e+00  9.730180e+08
75%    4.476000e+03  2.770000e+03  4.000000e+00  9.752209e+08
max    6.040000e+03  3.952000e+03  5.000000e+00  1.046455e+09

2D_matrix shape (6040, 3706)

[[5. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [3. 0. 0. ... 0. 0. 0.]]

6040 users
3706 items
Sparsity: 95.53%

Test and Train arrays are empty!



PD-R(ead) critics data from file?, 
PD-RML100(ead) ml100K data from file?, 
PD-RML1M(ead) ml1M data from file?, 
T(est/train datasets?, 
MF-ALS(atrix factorization- Alternating Least Squares)? 
MF-SGD(atrix factorization- Stochastic Gradient Descent)? 
===>> t

Generate both test and train data? Y or y, N or n: y

Train info: 6040 rows, 3706 cols
Test info: 6040 rows, 3706 cols
test ratings count = 187846
train ratings count = 812363
test + train count 1000209
test/train percentages: 18.78 / 81.22


Test and Train arrays are ready!



PD-R(ead) critics data from file?, 
PD-RML100(ead) ml100K data from file?, 
PD-RML1M(ead) ml1M data from file?, 
T(est/train datasets?, 
MF-ALS(atrix factorization- Alternating Least Squares)? 
MF-SGD(atrix factorization- Stochastic Gradient Descent)? 
===>> mf-als

Sample for ml-1M .. 

ALS instance parameters:
n_factors=20, user_reg=0.10000,  item_reg=0.10000, num_iters=10

[20,0.1,10]

Y or y to use these parameters or Enter to modify: y

Runtime parameters:
n_factors=20, user_reg=0.10000, item_reg=0.10000, max_iters=10, 
ratings matrix: 6040 users X 3706 items

Elapsed train/test time 0.00 secs
Iteration: 1
Train mse: 0.7104303284988095
Test mse: 1.8159974360128208
Elapsed train/test time 27.77 secs
Iteration: 2
Train mse: 0.578446125283257
Test mse: 1.6363680567251555
Elapsed train/test time 56.06 secs
Iteration: 5
Train mse: 0.48913115453293665
Test mse: 1.5326932290092146
Elapsed train/test time 86.36 secs
Iteration: 10
Train mse: 0.4619885262581458
Test mse: 1.5130742660401637
Elapsed train/test time 118.36 secs


PD-R(ead) critics data from file?, 
PD-RML100(ead) ml100K data from file?, 
PD-RML1M(ead) ml1M data from file?, 
T(est/train datasets?, 
MF-ALS(atrix factorization- Alternating Least Squares)? 
MF-SGD(atrix factorization- Stochastic Gradient Descent)? 
===>> mf-sgd
Sample for ml-1m .. 

SGD instance parameters:
num_factors K=20, learn_rate alpha=0.10000, reg beta=0.10000, num_iters=10, sgd_random=False

[20, 0.1, 0.1, 10]

Y or y to use these parameters or Enter to modify: y

Runtime parameters:
n_factors=20, learning_rate alpha=0.100, reg beta=0.10000, max_iters=10, sgd_random=False 
ratings matrix: 6040 users X 3706 items

Elapsed train/test time 0.00 secs
Iteration: 1
Train mse: 0.8673913396596189
Test mse: 0.9260681164880445
Elapsed train/test time 50.54 secs
Iteration: 2
Train mse: 0.8399165893244913
Test mse: 0.9110507708008525
Elapsed train/test time 102.44 secs
Iteration: 5
Train mse: 0.8064813447516811
Test mse: 0.8957664665203889
Elapsed train/test time 181.59 secs
Iteration: 10
Train mse: 0.7900410377005174
Test mse: 0.892978205265063
Elapsed train/test time 288.58 secs


PD-R(ead) critics data from file?, 
PD-RML100(ead) ml100K data from file?, 
PD-RML1M(ead) ml1M data from file?, 
T(est/train datasets?, 
MF-ALS(atrix factorization- Alternating Least Squares)? 
MF-SGD(atrix factorization- Stochastic Gradient Descent)? 
===>> 

Goodbye!


'''