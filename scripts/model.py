import pandas as pd
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import tree, ensemble
import xgboost as xgb
import lightgbm as gbm


if __name__ == '__main__':
    train = pd.read_csv('../Supermarket/Data/cleaned_train.csv')
    test = pd.read_csv('../Supermarket/Data/cleaned_test.csv')
    train = train[train['Open'] != 0]


    # def Store_to_train():
    #     #Use only store that are open to train
    #     train = train[train['Open'] != 0]
    #
    #     return train

    # def separate_X_y():
    #     train = Store_to_train()
    #     X = train.drop(['Sales', 'Customers'], axis = 1)
    #     y = train.Sales
    #
    #     return X, y


    #Split data to X and y using the function above
    # X, y = separate_X_y()

    def Scaler():
        scaler = preprocessing.StandardScaler()

        return scaler

    def rf_model():
        rf = ensemble.RandomForestRegressor(n_jobs=-1, n_estimators=15)

        return rf

    def xgb_model():
        model_xgb = xgb.XGBRegressor()

        return model_xgb

    def gbm_model():
        model_gbm = gbm.LGBMRegressor()

        return model_gbm

    #Fucntion call for store that open
    # train = Store_to_train()


    #build pipeline for scaling and building model
    def model_pipe_rf(X, y):
        my_pipe = pipeline.Pipeline(
            [('Scaling', Scaler()),
             ('Random_forest', rf_model())
             ])
        return my_pipe.fit(X,y)


    X = train.drop(['Sales', 'Customers'], axis=1)
    y = train.Sales

    #run pipeline
    model_pipe_rf(X,y)












