import pandas as pd
import matplotlib.pyplot as plt
from .geo import get_gps_lat_long_from_easting_northing
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler,FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
import os
import numpy as np

__all__ = ['FindMedianPriceModel']

class FindMedianPriceModel():
    """ Class to find and train the best model for median price"""
    
    def feature_cross(self,lat_df,lon_df):
        def latlon2string(lat,lon):
            combine=[]
            for i in lat:
                for j in lon:
                    combine.append(f"{i}_{j}")
            return pd.DataFrame(np.array(combine).T)
        
        def latlon2string_data(lat,lon):
            combine=[]
            for i, j in zip(lat,lon):
                    combine.append(f"{i}_{j}")
            return pd.DataFrame(np.array(combine).T)
        
        lat_=list(np.arange(48,62))
        lon_=list(np.arange(-10,4))
        corpus = latlon2string(lat_,lon_)
        ohe=OneHotEncoder().fit(corpus)
        colname=[f"lat_lon_{i}" for i in range(corpus.shape[0])]
        tmp=latlon2string_data(lat_df,lon_df)
        return pd.DataFrame(ohe.transform(tmp).toarray(),columns=colname)
        
         
    def load_data(self):
        """load all the features for median_price prediction model
        Returns
        -------
        pd.DataFrame
            data contains features related to total property value
        """
        # sampled_file = os.sep.join((os.getcwd(),
        #                             'resources',
        #                             'postcodes_sampled.csv'))
        # sector_file = os.sep.join((os.getcwd(),
        #                         'resources',
        #                         'households_per_sector.csv'))
        
        sampled_data=pd.read_csv('resources/postcodes_sampled.csv')
        
        df=sampled_data[["easting","northing","medianPrice","soilType","localAuthority","altitude"]]
        self.df=df
        return self
    
    def preprocessing(self):
        """preprocess the input data
        Returns
        -------
        pd.DataFrame
            data contains selected features related to total property value
        """
        df_=self.df.drop_duplicates()
        lat,lon=get_gps_lat_long_from_easting_northing(list(self.df["easting"]),list(self.df["northing"]))
        lat=pd.DataFrame(lat)
        lon=pd.DataFrame(lon)
        # latlon=self.feature_cross(lat, lon)
        df_=pd.concat([df_,lat,lon], axis=1,join="inner")
        df_=df_.drop(columns=["easting","northing"])
        df_.columns = ["medianPrice","soilType","localAuthority","altitude","Latitude","Longitude"]

        self.y=df_["medianPrice"]
        self.X=df_.drop(columns="medianPrice")
        
        self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.2)
        
        self.preproc=make_column_transformer(
            (OrdinalEncoder(),["soilType","localAuthority"]),
            (MinMaxScaler(),["altitude"]),
            # ("drop",["soilType","altitude"]),
            remainder="passthrough")
        
        return self

    def re_trainset(self):
        df_=self.df.drop_duplicates()
        lat,lon=get_gps_lat_long_from_easting_northing(list(self.df["easting"]),list(self.df["northing"]))
        lat=pd.DataFrame(lat.astype(int))
        lon=pd.DataFrame(lon.astype(int))
        # latlon=self.feature_cross(lat, lon)
        df_=pd.concat([df_,lat,lon], axis=1,join="inner")
        df_=df_.drop(columns=["easting","northing"])
        self.y=df_["medianPrice"]
        self.X=df_.drop(columns="medianPrice")
        return self.X, self.y

    def model_selection(self):
        """find the best model for regression prediction
        Returns
        -------
        type: sklearn.ensemble._forest.RandomForestRegressor
            it means the best_regressor
        """
        regression_model = {'regressor':(LinearRegression(),SGDRegressor(),KNeighborsRegressor(),SVR(),Lasso(),RandomForestRegressor())}
        model_=Pipeline([
            ("preprocessing",self.preproc),
            ("regressor",(LinearRegression(),SGDRegressor(),KNeighborsRegressor(),SVR(),Lasso(),RandomForestRegressor()))
        ])
        regression_search = GridSearchCV(model_,param_grid=regression_model,n_jobs=-1,scoring="neg_root_mean_squared_error")
        regression_search.fit(self.X_train,self.y_train)
        return regression_search.best_params_['regressor']
        

    def model_1(self):
        """use KNN model for prediction
        Returns
        -------
        test.best_estimator_, test.best_params_, test.best_score_
        
        """
        model_=Pipeline([
            ("preprocessing",self.preproc),
            ("regressor",KNeighborsRegressor())
        ])
        # model_.fit(self.X_train,self.y_train)
        test=RandomizedSearchCV(model_,{
            "regressor__n_neighbors":stats.randint(1,100),
            "regressor__weights":["distance"],
            "regressor__p":[1,2],
        },scoring="neg_root_mean_squared_error",cv=5)
        test.fit(self.X_train,self.y_train)
        print(self.y_test.mean())
        return test.best_estimator_, test.best_params_, test.best_score_

    def model_2(self):
        """use randomforest model for prediction
        Returns
        -------
        test.best_estimator_, test.best_params_, test.best_score_
        
        """
        model_=Pipeline([
            ("preprocessing",self.preproc),
            ("regressor",RandomForestRegressor())
        ])

        n_estimators = [int(x) for x in np.linspace(start = 1, stop = 20, num = 10)]
        max_features = ['sqrt']
        max_depth = [int(x) for x in np.linspace(10, 20, num = 2)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]

        random_grid = {'regressor__n_estimators': n_estimators,
               'regressor__max_features': max_features,
               'regressor__max_depth': max_depth,
               'regressor__min_samples_split': min_samples_split,
               'regressor__min_samples_leaf': min_samples_leaf,
               'regressor__bootstrap': bootstrap}
        
        test = RandomizedSearchCV(estimator = model_, param_distributions = random_grid, n_iter = 100, cv = 2, verbose=2, random_state=42, n_jobs = -1, scoring="neg_root_mean_squared_error")# Fit the random search model
        # test=RandomizedSearchCV(model_,{
        #     "n_estimators":stats.randint(1,200),
        #     "max_depth":[int(x) for x in np.linspace(10,88,num=10)],
        # },scoring="neg_root_mean_squared_error",cv=2,n_jobs=-1,verbose=2)
        test.fit(self.X_train,self.y_train)
        print(self.y_test.mean())
        return test.best_estimator_, test.best_params_, test.best_score_

    def predict(self,test_data):
        model_1 = KNeighborsRegressor(n_neighbors=80,p=2,weights='distance')
        model_2 = RandomForestRegressor(n_estimators=17, min_samples_split=10, min_samples_leaf=4, max_features='auto')
        model_1.fit(self.X_train,self.y_train)
        prediction = model_1.predict(test_data)
        return prediction