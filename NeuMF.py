import numpy as np
import pandas as pd
import os
import warnings
import math
import tensorflow as tf
from keras.models import load_model
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate, Dropout, BatchNormalization
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt
from sympy import evaluate
warnings.filterwarnings('ignore')



class NeuMF():
    def __init__(self, 
                dataset, 
                test_size=0.2,
                n_factors=5, 
                lr=1e-3, 
                n_layers=3, 
                n_nodes_per_layer=[],
                n_epochs=5, 
                batch_size=256, 
                model_num=1, 
                dropout_prob=0.2):
        
        self.dataset = dataset 
        self.cwd=os.getcwd() 
        self.n_users = len(pd.unique(dataset["user_id"]))
        self.n_items = len(pd.unique(dataset["item_id"]))
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.batch_size = batch_size 
        self.model_num = model_num
        self.dropout_prob = dropout_prob
        self.lr = lr
        self.n_layers = n_layers
        self.test_size = test_size
        self.n_nodes_per_layer = n_nodes_per_layer
        self.drop_out_prob = dropout_prob
    def train(self):
        # creating movie embedding path
        self.train, self.test, self.val = self.test_train_split()

        self.movie_input = Input(shape=[1], name="Movie-Input")
        self.movie_embedding = Embedding(self.n_items+1, self.n_factors, name="Movie-Embedding")(self.movie_input)
        self.movie_vec = Flatten(name="Flatten-Books")(self.movie_embedding)

        # creating user embedding path
        self.user_input = Input(shape=[1], name="User-Input")
        self.user_embedding = Embedding(self.n_users+1, self.n_factors, name="User-Embedding")(self.user_input)
        self.user_vec = Flatten(name="Flatten-Users")(self.user_embedding)

        # concatenate features
        conc = Concatenate()([self.movie_vec, self.user_vec])
        # add fully-connected-layers

        dense = Dense(self.n_nodes_per_layer[0], activation='relu')(conc)
        dropout = Dropout(self.dropout_prob)(dense)
        batch_norm = BatchNormalization()(dropout)

        for k, n_nodes in enumerate(self.n_nodes_per_layer[1:-1]):
            dense = Dense(n_nodes, activation='relu')(batch_norm)
            dropout = Dropout(self.dropout_prob)(dense)
            batch_norm = BatchNormalization()(dropout)
        
        dense = Dense(self.n_nodes_per_layer[-1], activation='relu')(batch_norm)
        out = Dense(1)(dense)

        self.model = Model([self.user_input, self.movie_input], out)
        self.model.compile(optimizer=Adam(learning_rate=self.lr), loss=MeanSquaredError())

    def test_train_split(self):
    
        train, temp_test = train_test_split(self.dataset, test_size=self.test_size, random_state=42)
        val, test = train_test_split(temp_test, test_size=0.5, random_state=42)

        return train, test, val
    def plot_learning_curve(self): 

        history = self.model.fit([self.train.user_id, self.train.item_id], 
                            self.train.rating, 
                            validation_data = ((self.val.user_id, self.val.item_id), self.val.rating), 
                            epochs=self.n_epochs, 
                            verbose=1, 
                            batch_size=self.batch_size)

        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        # plt.plot(train_loss, label='train')
        # plt.plot(val_loss, label ='val')
        # plt.yscale('log')
        # plt.ylabel('mse loss')
        # plt.xlabel('epochs')
        # plt.title(f'Model {self.model_num}: Loss Curves')
        # plt.legend()
        # plt.savefig(f'{self.cwd}/MLP_tunning/model_{self.model_num}_loss.png')
        # plt.close()

    def test_eval(self):

        predictions = self.model.predict([self.test.user_id, self.test.item_id])
        std = np.std(predictions)
        preds_list = []
        for rating in predictions: 
            preds_list.append(rating[0])
        
        pred_ratings = np.array(preds_list).astype('float64')
        actual_ratings = np.array(self.test.rating)

        test_mse = mean_squared_error(actual_ratings, pred_ratings)

        header = ['model', 'test mse', 'test std', 'epochs', 'lr', 'n_nodes_per_layer', 'n_factors', 'batch_size', 'dropout_prob']
        results = {'model':self.model_num, 
                'test mse': test_mse, 
                'test std':std,
                'epochs': self.n_epochs, 
                'lr':self.lr, 
                'n_nodes_per_layer': self.n_nodes_per_layer, 
                'n_factors':self.n_factors, 
                'batch_size':self.batch_size, 
                'dropout_prob':self.dropout_prob}
       

        results_df = pd.DataFrame.from_dict(results)
        results_df.to_csv(f'{self.cwd}/MLP_tunning/model_{self.model_num}_results.csv')
        return results

    def predict(self, u, i, id_to_movie):
        all_items = pd.unique(self.dataset["item_id"])
        user_array = np.asarray([u for i in range(len(all_items))])
        
        prediction = self.model.predict([user_array, all_items])

        item_index = np.where(all_items==i)

        rating=prediction[item_index[0]]
        item = id_to_movie[str(i)]
        print(rating[0][0], item)

        return rating, item

    def get_recommendations(self, u, id_to_movies):
        all_items = pd.unique(self.dataset["item_id"])
        user_array = np.asarray([u for i in range(len(all_items))])
        
        predictions = self.model.predict([user_array, all_items])
        preds = []
        for i in range(len(predictions)): 
            preds.append((predictions[i][0], id_to_movies[str(all_items[i])]))
            
        preds.sort(reverse=True)

        return preds

    def eval_recs(self): 

        predictions = self.model.predict([self.dataset.user_id, self.dataset.item_id])

        mse_error_list, mae_error_list = [], [] 
        for i in range(len(predictions)): 
            pred = predictions[i]
            actual = self.dataset.rating.iloc[i]
            if actual != 0:
                se = (pred - actual)**2
                error = abs(pred-actual)
                mse_error_list.append(se)
                mae_error_list.append(error)

        mse = sum(mse_error_list)/len(mse_error_list)
        mae = sum(mae_error_list)/len(mae_error_list)
        rmse = math.sqrt(mse)
        
        print("NeuMF MSE =  %f " % (mse))
        print("NeuMF MAE = %f " % (mae))
        print("NeuMF RMSE = %f " % (rmse))
        
        return mse, mae, rmse, mse_error_list

def ncf_read_data(filename):
    header = ['user_id','item_id','rating','timestamp']
    dataset = pd.read_csv(filename,sep = '\t',names = header)
    print(dataset.head())
    return dataset

    
# def get_recommendations_MLP(model, test, train, user):
#     #predictions = model.predict([test.user_id.head(10), test.item_id.head(10)])
#     predictions = model.predict([test.user_id, test.item_id])
#     #print("Predictions for user %d: " % user)
    
#     # [print(predictions[i], test.rating.iloc[i]) for i in range(0,10)]   
#     return predictions

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

def ncf_grid_search(dataset):
    model_num=1
    n_factors = [5,25,50,100]
    n_nodes_per_layer = [128, 64, 32, 16, 8, 4, 2]
    lrs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    dropout_prob = 0.2
    n_epochs = 25 
    batch_size=256 
    model_num = 0
    headers =  ['model', 'test mse', 'test std', 'epochs', 'lr', 'n_nodes_per_layer', 'n_factors', 'batch_size', 'dropout_prob']
    all_results = pd.DataFrame(columns=headers)
    for factors in n_factors: 
        for lr in lrs: 
            MLP = NeuMF(dataset, 
                test_size=0.2,
                n_factors=factors, 
                lr=lr, 
                n_layers=3, 
                n_nodes_per_layer=n_nodes_per_layer,
                n_epochs=n_epochs, 
                batch_size=batch_size, 
                model_num=model_num, 
                dropout_prob=dropout_prob)
            model_num += 1

            MLP.train()

            MLP.plot_learning_curve()

            results = MLP.test_eval()
            all_results = all_results.append(results, ignore_index=True)
    all_results.to_csv("MLP_tunning_results.csv")

def main():
    cwd = os.getcwd() 

    path = cwd + "/data/ml-100k/u.data"
    dataset = ncf_read_data(path)

    ncf_grid_search(dataset)
    file_dir = 'data/ml-100k/' # path from current directory
    datafile = 'u.data'  # ratings file
    itemfile = 'u.item'  # movie titles file    
    genrefile = 'u.genre' # movie genre file     
    movies, genres, features = from_file_to_2D(cwd, file_dir+genrefile, file_dir+itemfile)

    

    # build neural net 
    MLP = NeuMF(dataset, 
                test_size=0.2,
                n_factors=5, 
                lr=1e-3, 
                n_layers=3, 
                n_nodes_per_layer=[128, 64, 32, 16, 8, 4, 2],
                n_epochs=10, 
                batch_size=256, 
                model_num=1, 
                dropout_prob=0.2)
    # train model 
    MLP.train()
    # make predictions 
    MLP.plot_learning_curve()

    # MLP.test_eval()
    # n = 10
    # preds = MLP.get_recommendations(340, movies)[:n]
   
    # for p in preds:
    #     print(p)
    # #MLP.predict(340, 1, movies)
    # MLP.eval_recs() 
    # user_id = 1
    # predictions = get_recommendations_MLP(model, test, train, user_id)
    # predictions.sort()
    # predictions.reverse()
    # n = 10
    # top_n_predictions = predictions[:n]
    # print(top_n_predictions)
    # mse, mae, rmse, len_error_list = evaluate_recs(predictions, test, user_id)

if __name__ == "__main__":
    main() 