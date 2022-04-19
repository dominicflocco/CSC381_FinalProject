import numpy as np
import pandas as pd
import os
import warnings
import math
from keras.models import load_model
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from keras.models import Model
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
                n_nods_per_layer=[],
                n_epochs=5, 
                batch_size=256, 
                model_num=1, 
                dropout_prob=0.2, 
                cwd=''):
        
        self.dataset = dataset 
        self.cwd=os.getcwd() 
        self.n_users = len(pd.unique(dataset["user_id"]))
        self.n_items = len(pd.unique(dataset["item_id"]))
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.batch_size = batch_size 
        self.model_num = model_num
        self.lr = lr
        self.n_layers = n_layers
        self.test_size = test_size
        self.n_nodes_per_layer = []
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
            dropout = Dropout(dropout_prob)(dense)
            batch_norm = BatchNormalization()(dropout)
        
        dense = Dense(self.n_nodes_per_layer[-1], activation='relu')(batch_norm)
        out = Dense(1)(dense)

        self.model = Model([self.user_input, self.movie_input], out)
        self.mode.compile(optimizer=Adam(learn_rate=self.lr, loss=MeanSquaredError()))

    def test_train_split(self):
    
        train, temp_test = train_test_split(self.dataset, test_size=self.test_size, random_state=42)
        val, test = train_test_split(temp_test, test_size=0.5, random_state=42)

        return train, test, val
    def plot_learning_curve(self): 
        history = self.model.fit([self.train.user_id, self.train.item_id], 
                            self.train.rating, 
                            validation_data = ((self.val.user_id, self.val.item_id), val.rating), 
                            epochs=self.n_epochs, 
                            verbose=1, 
                            batch_size=self.batch_size)

        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.plot(train_loss, label='train')
        plt.plot(val_loss, label ='val')
        plt.yscale('log')
        plt.ylabel('mse loss')
        plt.xlabel('epochs')
        plt.title(f'Model {self.model_num}: Loss Curves')
        plt.legend()
        plt.savefig(f'{self.cwd}/MLP_tunning/model_{self.model_num}_loss.png')

    def eval(self):

        predictions = self.model.predict([self.test.user_id, self.test.movie_id])
        std = np.std(predictions)
        preds_list = []
        for rating in predictions: 
            preds_list.append(rating[0])
        
        pred_ratings = np.array(preds_list).astype('float64')
        actual_ratings = np.array(self.test.rating)

        test_mse = mean_squared_error(actual_ratings, pred_ratings)

        header = ['model', 'test mse', 'test std', 'epochs', 'lr', 'n_nodes_per_layer', 'n_factors', 'batch_size', 'dropout_prob']
        results = [[self.model_num, test_mse, std, self.n_epochs, self.lr, self.n_nodes_per_layer, self.n_factors, self.batch_size, self.dropout_prob]]

        results_df = pd.DataFrame(results, columns=header)
        results_df.to_csv(f'{self.cwd}/MLP_tunning/model_{self.model_num}_results.csv')
        return results

def read_data(filename):
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

def evaluate_recs(predictions, test, user): 

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

def main():
    cwd = os.getcwd() 
    path = cwd + "/data/ml-100k/u.data"
    dataset = read_data(path)

    model_num=1
    n_factors = [5,25,50,100]
    n_nodes_per_layer = [128, 64, 32, 16, 8, 4, 2]
    lrs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    dropout_prob = 0.2
    n_epochs = 25 
    batch_size=256 

    # build neural net 
    MLP = NeuMF(dataset, 
                test_size=0.2,
                n_factors=5, 
                lr=1e-3, 
                n_layers=3, 
                n_nods_per_layer=[128, 64, 32, 16, 8, 4, 2],
                n_epochs=10, 
                batch_size=256, 
                model_num=1, 
                drop_out_prob=0.2, 
                cwd='')
    # train model 
    MLP.train()
    # make predictions 
    MLP.plot_learning_curve()

    MLP.eval()
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