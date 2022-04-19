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
from sympy import evaluate
warnings.filterwarnings('ignore')

class NeuMF():
    def __init__(self, 
                dataset, 
                test_size=0.2,
                n_factors=5, 
                lr=1e-3, 
                n_layers=3):
        
        self.dataset = dataset 
        
        self.n_users, self.n_items = len(pd.unique(dataset["user_id"])), len(pd.unique(dataset["item_id"]))
        self.n_factors = n_factors
        self.lr = lr
        self.n_layers = n_layers
        self.test_size = test_size
        self.model = None
        
    def build_model()
    

def read_data(filename):
    header = ['user_id','item_id','rating','timestamp']
    dataset = pd.read_csv(filename,sep = '\t',names = header)
    print(dataset.head())
    return dataset

def build_MLP(dataset, train, test):
    # creating book embedding path
    n_users = len(pd.unique(dataset["user_id"]))
    n_items = len(pd.unique(dataset["item_id"]))
    
    movie_input = Input(shape=[1], name="Movie-Input")
    movie_embedding = Embedding(n_items+1, n_factors, name="Movie-Embedding")(movie_input)
    movie_vec = Flatten(name="Flatten-Books")(movie_embedding)
    # creating user embedding path
    user_input = Input(shape=[1], name="User-Input")
    user_embedding = Embedding(n_users+1, n_factors, name="User-Embedding")(user_input)
    user_vec = Flatten(name="Flatten-Users")(user_embedding)
    # concatenate features
    conc = Concatenate()([movie_vec, user_vec])
    # add fully-connected-layers
    
    fc1 = Dense(128, activation='relu')(conc)
    fc2 = Dense(32, activation='relu')(fc1)
    out = Dense(1)(fc2)
    # Create model and compile it
    model2 = Model([user_input, movie_input], out)
    compile(optimizer=Adam(learning_rate=lr), loss=MeanSquaredError())

    model2.compile('adam', 'mean_squared_error')

    return model2  
    
def get_recommendations_MLP(model, test, train, user):
    #predictions = model.predict([test.user_id.head(10), test.item_id.head(10)])
    predictions = model.predict([test.user_id, test.item_id])
    #print("Predictions for user %d: " % user)
    
    # [print(predictions[i], test.rating.iloc[i]) for i in range(0,10)]   
    return predictions

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

    # train-test split 
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)
    # build neural net 
    model = build_MLP(dataset, train, test)
    # train model 
    history = model.fit([train.user_id, train.item_id], train.rating, epochs=5, verbose=1)
    # make predictions 
    user_id = 1
    predictions = get_recommendations_MLP(model, test, train, user_id)
    predictions.sort()
    predictions.reverse()
    n = 10
    top_n_predictions = predictions[:n]
    print(top_n_predictions)
    mse, mae, rmse, len_error_list = evaluate_recs(predictions, test, user_id)

if __name__ == "__main__":
    main() 