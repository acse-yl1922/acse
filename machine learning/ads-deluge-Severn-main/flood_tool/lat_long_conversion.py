from sklearn.base import TransformerMixin, BaseEstimator
from .geo import get_gps_lat_long_from_easting_northing
import numpy as np 
import pandas as pd

__all__ = ['CustomScaler']

class CustomScaler(TransformerMixin, BaseEstimator): 
    """ Class to convert from easting/northing to latitude/longitude"""
    def __init__(self):
        return None
    
    def fit(self, X, y=None):
        """ 
        Parameters
        ----------
        X: DataFrame 
           easting and northing values 
           Features of the model 
        y: DataFrame/Series
           (optional) Label. 
        Returns
        -------
        self 
        """
        return self
    
    def transform(self, X, y=None):
        """ 
        Parameters
        ----------
        X: DataFrame 
           easting and northing values 
           Features of the model 
        y: DataFrame/Series
           (optional) Label. 
        Returns
        -------
        DataFrame 
            latitude and Longitude values 
        """
        self.easting = np.array(X['easting'])
        self.northing = np.array(X['northing'])
        self.results = get_gps_lat_long_from_easting_northing(self.easting, self.northing)
        self.latitude = self.results[0]
        self.longitude = self.results[1]
        return pd.DataFrame({'latitude': self.latitude, 'longitude': self.longitude})