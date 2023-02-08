import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler,FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from .geo import get_gps_lat_long_from_easting_northing
from sklearn.pipeline import Pipeline

__all__ = ['MedianPriceModel']

class MedianPriceModel():
    """ Class to predict the median house price """

    def __init__(self, data):
        """
        Constructs the necessary attrbute for the class/object
        
        Parameters
        ----------
        data : pandas DataFrame
                 training_data containing features and label.
        """
        self.data = data

    def train(self):
        """This method preprocesses the data, separates the features and label,
        creates and trains a pipeline model for predictions
        
        Returns
        -------
        model: Pipeline
               fully trained pipeline model
        """
        X = self.data[["soilType","localAuthority","altitude"]]
        y = self.data["medianPrice"]
        
        lat,lon=get_gps_lat_long_from_easting_northing(list(self.data["easting"]),list(self.data["northing"]))
        X["latitude"] = lat 
        X["longitude"] = lon

        preproc = make_column_transformer(
            (OrdinalEncoder(),["soilType","localAuthority"]),
            (MinMaxScaler(),["altitude"]),
            remainder="passthrough")   

        model=Pipeline([
            ("preprocessing",preproc),
            ("regressor",KNeighborsRegressor(n_neighbors=23, p=1, weights="distance"))
        ])

        model.fit(X, y)

        return model

    
    def predict(self, model, test_data):
        """ A method to predict the median price for unseen data with the pipeline model 
        
        Returns
        -------
        prediction: numpy.array
                 median house price for different locations 
        """
        
        prediction = model.predict(test_data)

        return prediction