import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from .geo import *
import pickle

__all__ = ['FloodClassModel']

class FloodClassModel():
    """ Class to predict the flood probability from given data """

    def __init__(self, training_data):
        """
        Constructs the necessary attrbute for the class/object
        
        Parameters
        ----------
        training_data : pandas DataFrame
                        training_data containing features and label.
        """
        self.training_data = training_data
        self.preproc = self.create_column_transformer()
        self.model_pipeline = self.create_model_pipeline()


    def train(self):
        """ A method to fit the model with the training data
        
        Returns
        -------
        Pipeline
             fitted model pipeline
        """    
        X, y = self.prepare_data_for_training()

        print('==== Fitting the data using random forest regressor')
        self.model_pipeline.fit(X, y)

        return self.model_pipeline


    def train_with_randomized_search(self):
        """ This method calls the following method to initiate randomized search for hyperparameters 
        >>optimize_rf_model(X, y)
        """
        
        X, y = self.prepare_data_for_training()

        self.model_pipeline = self.optimize_rf_model(X, y)


    def prepare_data_for_training(self):
        """This method basically converts the flood probability classification label into a regression label using a dictionary.
        The conversion of the label data is performed after the data goes through full_preprocessing.
        
        Return
        ------
        X: pandas DataFrame
           Preprocessed Dataframe for the features of the model
        y: pandas Series
           the converted label of the data from classification to regression 
        """
        X, y = self.perform_full_preprocess()

        label_conv_dict = {1: 0.01, 2: 0.5, 3: 1, 4:5, 5:10, 6:15, 7:20, 8:30, 9:40, 10:50}
        y = y.apply(lambda y: label_conv_dict[y])
        return X, y
        
    
    def predict(self, unlabelled_data):
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
        print('==== Generating prediction for flood class from postcodes')
        y_pred = self.model_pipeline.predict(unlabelled_data)

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
        flood_event = np.array([0.01,0.5,1,5,10,15,20,30,40,50])
        distances = abs(flood_event - flood_prob)
        flood_class = int(np.argmin(distances) + 1)
        return flood_class


    def create_model_pipeline(self):
        """A method that creates a pipeline combining the preprocessing ColumTransformer with a model
        
        Returns
        -------
        model_pipe: Pipeline
               final Pipeline containing optimum model for prediction
        """
        model_pipe = Pipeline([
            ('preprocessing', self.preproc),
            ('classifier', RandomForestRegressor(max_depth=27, min_samples_leaf=2, n_estimators=424))
        ])

        return model_pipe


    def optimize_rf_model(self, X, y):
        """Hyperparameter search to optimize model/pipeline
        
        Parameters
        ----------
        X: pandas DataFrame 
           The features of the training data 
        y: pandas Series
           The label of the training data
           
        Returns
        -------
        best estimator: Pipeline
                   The trained pipeline/model with the best hyperparameters
        """
        n_estimators = [int(x) for x in np.linspace(start = 100, stop = 600, num = 40)]
        max_depth = [int(x) for x in np.linspace(10, 35, num = 8)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]

        random_grid = {
            'classifier__n_estimators': n_estimators,
            'classifier__max_depth': max_depth,
            'classifier__min_samples_split': min_samples_split,
            'classifier__min_samples_leaf': min_samples_leaf,
            'classifier__bootstrap': bootstrap
        }

        final_search = RandomizedSearchCV(self.model_pipeline,
                            param_distributions=random_grid,
                            n_iter=2,
                            random_state=42,
                            cv = 2,
                            verbose=2,
                            n_jobs=-1,
                            scoring='r2')

        final_search.fit(X, y)

        pickle.dump(final_search.best_estimator_, open('flood_class_pipeline_real_testing.pkl','wb'));

        return final_search.best_estimator_


    def perform_full_preprocess(self):
        """This method performs preprocessing of the data by droping duplicates and calling 2 methods from this module. 
        >>convert_east_northing_to_lat_long() to convert the locations 
        >>separate_data() to separate features from label 
        
        Returns
        -------
        X: pandas DataFrame
           Preprocessed Dataframe for the features of the model
        y: pandas Series
           the label of the data
        """
        self.training_data = self.training_data.drop_duplicates()

        print('==== Converting easting northing to lat long')
        self.convert_east_northing_to_lat_long()

        X, y = self.separate_data()

        return X, y
        
    
    def create_column_transformer(self):
        """A method to create a column_transformer from sklearn, for standard processing of the categorical
        and numerical features, prior to being trained in a model. 
        
        Returns
        -------
        ColumnTransformer
        preprocessor containing:
             -numerical columns: SimpleImputer() and StandardScaler()
             -categorical column: SImpleImputer(strategy=most_frequent) and OneHotEncoder(handle_unknown=ignore) 
        """
        num_pipe = Pipeline([
             ('num_imputer', SimpleImputer()),('num_scaler', StandardScaler())
        ])
    
        cat_pipe = Pipeline([
            ('cat_imputer',SimpleImputer(strategy = 'most_frequent')),
            ('cat_encoder',OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])

        preprocessor = ColumnTransformer([
            ('num_transformer', num_pipe, ['altitude']),
            ('cat_transformer', cat_pipe, ['localAuthority']),
            ('drop', 'drop', ['postcode', 'sector', 'soilType'])
        ], remainder='passthrough')

        return preprocessor



    def convert_east_northing_to_lat_long(self):
        """A method that converts the location on the data from easting and northing to
        latitude and longitude. it does this by calling the function below from the module 'geo.py' 
        to do the conversion. 
        
        >>get_gps_lat_long_from_easting_northing(easting, northing)
        
        """
        easting, northing = np.array(self.training_data['easting']), np.array(self.training_data['northing'])
        lat, long = get_gps_lat_long_from_easting_northing(easting, northing)

        self.training_data['latitude'] = lat
        self.training_data['longitude'] = long
        self.training_data = self.training_data.drop(columns=['easting', 'northing'])


    def separate_data(self):
        """This method separates features from label in a given data
        
        Returns
        -------
        X: pandas DataFrame
           the features of the data
        y: pandas Series
           the label of the data (riskLabel) 
        """
        X = self.training_data.drop(columns=['riskLabel','medianPrice'])
        y = self.training_data.riskLabel

        return X, y
    
