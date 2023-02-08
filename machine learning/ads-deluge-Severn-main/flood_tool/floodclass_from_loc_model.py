from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from .lat_long_conversion import CustomScaler
import numpy as np
import pandas as pd

__all__ = ['FloodClassFromLocationModel']

class FloodClassFromLocationModel():
    """ Class to predict the flood probability from location (easting/northing) alone """

    def __init__(self, data):
        """
        Constructs the necessary attrbute for the class/object
        
        Parameters
        ----------
        data : pandas DataFrame
               training_data containing locations and flood risk label.
        """
        self.data = data

    def train(self, lat_long=False):
        """This method separates the features and preprocesses the data in a pipeline. 
        It converts the location from easting/northing to latitude/longitude using a custom transformer in a pipeline. 
        Finally, it returns a fully fitted pipeline ready for predictions
        
        Returns
        -------
        pipe: Pipeline
               fully trained pipeline 
        """
        X = self.data[['easting', 'northing']]
        y = self.data['riskLabel']

        pipe = Pipeline([
            ('loc_conversion', CustomScaler()),
            ('num_imputer', SimpleImputer(strategy='median')), 
            ('num_scaler', StandardScaler()), 
            ('classifier', RandomForestRegressor(max_depth=27, min_samples_leaf=2, n_estimators=100))
        ])

        if lat_long:
            pipe = Pipeline([
                ('num_imputer', SimpleImputer(strategy='median')), 
                ('num_scaler', StandardScaler()), 
                ('classifier', RandomForestRegressor(max_depth=27, min_samples_leaf=2, n_estimators=100))
            ])
       
        print("====> fitting the flood class from location model")
        
        pipe.fit(X, y)
        return pipe


    def predict(self, model, unlabelled_data):
        """A method for model prediction of unlabelled data set. The predictions are converted back into a classification data
        using the method >>classify_y_pred()
        
        Parameters
        ----------
        unlabelled_data: pandas Series 
                     data containing features used for prediction
        Returns
        -------
        List
          fload probability predictions as a classification data
        """
        print("====> generating predictions for flood class from location")
        y_pred = model.predict(unlabelled_data)

        predictions = pd.Series(y_pred)
        predictions_converted = predictions.apply(lambda y: self.classify_y_pred(y))
       
        return list(predictions_converted)


    def classify_y_pred(self, flood_prob):
        """Conversion of flood probability predictions from a continuous data back to classification data
        
        Parameters
        ----------
        flood_prob: numpy.array
                predicted flood probability from model (continuous data)
        Returns
        -------
        numpy.array
           flood class probability 
        """       
        if flood_prob > 10:
            return 10
        if flood_prob < 1:
            return 1
            
        return int(round(flood_prob))