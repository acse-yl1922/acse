import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

__all__ = ['LocalAuthorityModel']

class LocalAuthorityModel():
    """ Class to predict the Local Authority given the easting and northing location"""

    def __init__(self, data):
        """
        Constructs the necessary attrbute for the class/object
        
        Parameters
        ----------
        data : pandas DataFrame
                 training_data containing features and label.
        """
        self.data = data
        self.encoder = LabelEncoder()

    def train(self):
        """This method separates the features from the label and encodes the label data. 
        It trains a KNeighborsClassifier model for predictions
        
        Returns
        -------
        best_knn_model: KNeighborsClassifier
             fully trained model for predicting local authority
        """

        X = self.data[["easting","northing"]]
        y = self.data["localAuthority"]

        self.encoder = LabelEncoder().fit(y)
        y = self.encoder.transform(y)

        best_knn_model = KNeighborsClassifier(n_neighbors = 1)
        best_knn_model.fit(X, y)

        return best_knn_model
    
    def predict(self, model, unlabelled_data):
        """ A method to predict the local authority for unseen data with the pipeline model. 
        The encoded labels are converted back to their original form after prediction
        
        Returns
        -------
        prediction: numpy.array
                 Local Authority for different locations 
        """
        y_pred = model.predict(unlabelled_data)
        prediction = self.encoder.inverse_transform(y_pred)
        
        return prediction