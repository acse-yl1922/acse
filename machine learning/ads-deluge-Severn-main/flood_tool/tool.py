import os

import numpy as np
import pandas as pd

from .geo import *
from .floodclassmodel import FloodClassModel
from .localauthoritymodel import LocalAuthorityModel
from .median_price_model import MedianPriceModel
from .floodclass_from_loc_model import FloodClassFromLocationModel

__all__ = ['Tool']


class Tool(object):
    """Class to interact with a postcode database file."""

    def __init__(self, postcode_file='', sample_labels='',
                 household_file=''):
        """Initializes flood and house data to be used by the models. 
        Initializes models for predicting flood class, median house price, and local authority.

        Parameters
        ----------

        postcode_file : str, optional
            Filename of a .csv file containing unlabelled geographic location
            data for postcodes.
        
        sample_labels: str, optional
            Filename of a .csv file containing labelled data for postcodes

        household_file : str, optional
            Filename of a .csv file containing information on households
            by postcode.
        """

        if postcode_file == '':
            full_postcode_file = os.sep.join((os.path.dirname(__file__),
                                         'resources',
                                         'postcodes_unlabelled.csv'))
        else:
            full_postcode_file = os.sep.join((os.path.dirname(__file__),
                                         'resources',
                                         postcode_file))

        if household_file == '':
            household_file = os.sep.join((os.path.dirname(__file__),
                                          'resources',
                                          'households_per_sector.csv'))
        else:
            household_file = os.sep.join((os.path.dirname(__file__),
                                          'resources',
                                          household_file))

        if sample_labels == '':
            sample_labels = os.sep.join((os.path.dirname(__file__),
                                         'resources',
                                         'postcodes_sampled.csv'))
        else:
            sample_labels = os.sep.join((os.path.dirname(__file__),
                                         'resources',
                                         sample_labels))


        self.postcode_sampled = pd.read_csv(sample_labels)
        self.postcodedb = pd.read_csv(full_postcode_file)
        self.household_data = pd.read_csv(household_file)

        self.floodClassModel = FloodClassModel(self.postcode_sampled)
        self.floodClassFromLocModel = FloodClassFromLocationModel(self.postcode_sampled)
        self.localAuthModel = LocalAuthorityModel(self.postcode_sampled)
        self.medianPriceModel = MedianPriceModel(self.postcode_sampled)

        self.flood_class_model_trained = None
        self.local_auth_model_trained = None
        self.median_price_model_trained = None
        self.flood_class_from_loc_model_trained = None

        self.flood_class_from_postcode_methods = {}
        self.local_auth_from_postcode_methods = {}
        self.median_price_from_postcode_methods = {}
        self.flood_class_from_loc_methods = {}
        

    def train(self):
        """
        Train the models using a labelled set of samples. 
        Trains two model for flood class, and one for local authority and median price

        """

        print('Training flood class model')
        self.flood_class_model_trained = self.floodClassModel.train()
        self.flood_class_from_postcode_methods[0] = self.flood_class_model_trained

        print('Training local auth model')
        self.local_auth_model_trained = self.localAuthModel.train()
        self.local_auth_from_postcode_methods[0] = self.local_auth_model_trained

        print('Training median price model')
        self.median_price_model_trained = self.medianPriceModel.train()
        self.median_price_from_postcode_methods[0] = self.local_auth_model_trained

        print('Training flood class from loc model')
        self.flood_class_from_loc_model_trained = self.floodClassFromLocModel.train()
        self.flood_class_from_loc_methods[0] = self.flood_class_from_loc_model_trained

    
    def train_flood_class_cv_optimization(self):
        """
        Train the flood class model using a labelled set of samples. 
        Optimize the hyperparameters of the model using a randomized search CV

        """
        print('Training flood class model')
        self.flood_class_model_trained = self.floodClassModel.train_with_randomized_search()
        self.flood_class_from_postcode_methods[0] = self.flood_class_model_trained

        print('Training local auth model')
        self.local_auth_model_trained = self.localAuthModel.train()
        self.local_auth_from_postcode_methods[0] = self.local_auth_model_trained

        print('Training median price model')
        self.median_price_model_trained = self.medianPriceModel.train()
        self.median_price_from_postcode_methods[0] = self.local_auth_model_trained

        print('Training flood class from loc model')
        self.flood_class_from_loc_model_trained = self.floodClassFromLocModel.train()
        self.flood_class_from_loc_methods[0] = self.flood_class_from_loc_model_trained



    def format_postcodes(self, postcodes):
        """Ensures that the postcodes given as input are converted to the correct format
        i.e. (XXXX XXX or XXX XXX)

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        Returns
        -------

        list
            list containing the same postcodes from the input in the correct format
         """

        pcodes = []
        for p in postcodes:
            p = p.strip()
            spl = p.split()
            if len(spl) == 1:
                unit = p[-3:]
                sector = p[:-3]
                pcodes.append(sector + " " + unit)
            else:
                pcodes.append(spl[0] + " " + spl[1])
        return pcodes


    def get_all_features_for_postcodes(self, postcodes):
        """Get a full list of features (latitude, longitude, localAuthority, 
        soilType, altitude) from a collection of input postcodes.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing latitude, longitude, localAuthority, 
            soilType, altitude indexed by the input postcodes. 
            Invalid postcodes (i.e. not in the input unlabelled postcodes file) 
            will raise an exception
         """
        postcodes = self.format_postcodes(postcodes)

        lat_long = self.get_lat_long(postcodes)

        frame = self.postcodedb.copy().set_index('postcode')
        try:
            feats = frame.loc[postcodes, ['localAuthority', 'soilType', 'altitude']]
        except:
            raise Exception('postcode not found in the unlabelled data, please use valid postcodes')

        return pd.concat([lat_long, feats], axis=1)


    def get_easting_northing(self, postcodes):
        """Get a frame of OS eastings and northings from a collection
        of input postcodes.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing only OSGB36 easthing and northing indexed
            by the input postcodes. Invalid postcodes (i.e. not in the
            input unlabelled postcodes file) raise an exception
         """

        postcodes = self.format_postcodes(postcodes)

        frame = self.postcodedb.copy()
        frame = frame.set_index('postcode')

        try:
            eastnorthdf = frame.loc[postcodes, ['easting', 'northing']]

        except:
            raise Exception("Cannot find postcode in unlabelled data, please use valid postcodes")

        return eastnorthdf
    

    def get_lat_long(self, postcodes):
        """Get a frame containing GPS latitude and longitude information for a
        collection of of postcodes.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing only WGS84 latitude and longitude pairs for
            the input postcodes. Invalid postcodes (i.e. not in the
            input unlabelled postcodes file) will raise an exception.
        """

        eastingnorthingdf = self.get_easting_northing(postcodes)
        lat, long = get_gps_lat_long_from_easting_northing(np.array(eastingnorthingdf['easting']), np.array(eastingnorthingdf['northing']))
        eastingnorthingdf['easting'] = lat
        eastingnorthingdf['northing'] = long
        eastingnorthingdf.rename({'easting': 'latitude', 'northing': 'longitude'}, axis=1, inplace=True)

        return eastingnorthingdf

    @staticmethod
    def get_flood_class_from_postcodes_methods():
        """
        Get a dictionary of available flood probablity classification methods
        for postcodes.

        Returns
        -------

        dict
            Dictionary mapping classification method names (which have
             no inate meaning) on to an identifier to be passed to the
             get_flood_class_from_postcode method.
        """
        
        return {0: 0}


    def get_flood_class_from_postcodes(self, postcodes, method=0):
        """
        Generate series predicting flood probability classification
        for a collection of poscodes using RandomForestRegression.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        method : int (optional)
            optionally specify (via a value in
            get_flood_class_from_postcodes_methods) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by postcodes.
        """

        if method != 0:
            return pd.Series(data=np.ones(len(postcodes), int),
                             index=np.asarray(postcodes),
                             name='riskLabel')
        else:

            model = self.flood_class_from_postcode_methods[method]

            test_data = self.get_all_features_for_postcodes(postcodes)
            # print(test_data)
        
            floodClassPredictions = model.predict(test_data)
            
            # self.flood_class_model_trained.predict(test_data)

            return pd.Series(floodClassPredictions, index=postcodes)


    @staticmethod
    def get_flood_class_from_locations_methods():
        """
        Get a dictionary of available flood probablity classification methods
        for locations.

        Returns
        -------

        dict
            Dictionary mapping classification method names (which have
             no inate meaning) on to an identifier to be passed to the
             get_flood_class_from_OSGB36_locations and
             get_flood_class_from_OSGB36_locations method.
        """
        return {0: 0}


    def get_flood_class_from_OSGB36_locations(self, eastings, northings, method=0):
        """
        Generate series predicting flood probability classification
        for a collection of OSGB36_locations.

        Parameters
        ----------

        eastings : sequence of floats
            Sequence of OSGB36 eastings.
        northings : sequence of floats
            Sequence of OSGB36 northings.
        method : int (optional)
            optionally specify (via a value in
            self.get_flood_class_from_locations_methods) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by locations.
        """

        if method != 0:
            return pd.Series(data=np.ones(len(eastings), int),
                             index=[(est, nth) for est, nth in
                                    zip(eastings, northings)],
                             name='riskLabel')
        else:
            test_data = pd.DataFrame({'easting': eastings, 'northing': northings})

            idx = zip(eastings, northings)

            model = self.flood_class_from_loc_model_trained
            predictions = self.floodClassFromLocModel.predict(model, test_data)

            return pd.Series(predictions, index=idx)


    def get_flood_class_from_WGS84_locations(self, longitudes, latitudes, method=0):
        """
        Generate series predicting flood probability classification
        for a collection of WGS84 datum locations.

        Parameters
        ----------

        longitudes : sequence of floats
            Sequence of WGS84 longitudes.
        latitudes : sequence of floats
            Sequence of WGS84 latitudes.
        method : int (optional)
            optionally specify (via a value in
            self.get_flood_class_from_locations_methods) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by locations.
        """

        if method != 0:
            return pd.Series(data=np.ones(len(longitudes), int),
                             index=[(lng, lat) for lng, lat in
                                    zip(longitudes, latitudes)],
                             name='riskLabel')
        else:
            test_data = pd.DataFrame({'latitude': latitudes, 'longitude': longitudes})
            
            idx = zip(latitudes, longitudes)

            model = self.floodClassFromLocModel.train(lat_long=True)
            predictions = self.floodClassFromLocModel.predict(model, test_data)

            return pd.Series(predictions, index=idx)

    @staticmethod
    def get_house_price_methods():
        """
        Get a dictionary of available flood house price regression methods.

        Returns
        -------

        dict
            Dictionary mapping regression method names (which have
             no inate meaning) on to an identifier to be passed to the
             get_median_house_price_estimate method.
        """
        return {0: 0}


    def get_median_house_price_estimate(self, postcodes, method=0):
        """
        Generate series predicting median house price for a collection
        of poscodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        method : int (optional)
            optionally specify (via a value in
            self.get_house_price_methods) the regression
            method to be used.

        Returns
        -------

        pandas.Series
            Series of median house price estim ates indexed by postcodes.
        """

        if method != 0:
            return pd.Series(data=np.full(len(postcodes), 245000.0),
                             index=np.asarray(postcodes),
                             name='medianPrice')
        else:
            test_data = self.get_all_features_for_postcodes(postcodes)

            predictions = self.median_price_model_trained.predict(test_data)
            return pd.Series(predictions, index=postcodes)

    @staticmethod
    def get_local_authority_methods():
        """
        Get a dictionary of available local authorithy classification methods.

        Returns
        -------

        dict
            Dictionary mapping regression method names (which have
             no inate meaning) on to an identifier to be passed to the
             get_altitude_estimate method.
        """
        
        return {0: 0}

    def get_local_authority_estimate(self, eastings, northings, method=0):
        """
        Generate series predicting local authorities for a sequence
        of OSGB36 locations.

        Parameters
        ----------

        eastingss : sequence of floats
            Sequence of OSGB36 eastings.
        northings : sequence of floats
            Sequence of OSGB36 northings.
        method : int (optional)
            optionally specify (via a value in
            self.get_altitude_methods) the regression
            method to be used.

        Returns
        -------

        pandas.Series
            Series of local authorities indexed by easting northing locations.
        """

        if method != 0:
            return pd.Series(data=np.full(len(eastings), 'Unknown'),
                             index=[(est, nth) for est, nth in
                                    zip(eastings, northings)],
                             name='localAuthority')
        else:
            test_data = pd.DataFrame({'easting': eastings, 'northing': northings})

            idx = zip(eastings, northings)

            model = self.local_auth_model_trained
            predictions = self.localAuthModel.predict(model, test_data)

            return pd.Series(predictions, index=idx)


    def get_total_value(self, postal_data):
        """
        Return a series of estimates of the total property values
        of a sequence of postcode units or postcode sectors.


        Parameters
        ----------

        postal_data : sequence of strs
            Sequence of postcode units or postcodesectors


        Returns
        -------

        pandas.Series
            Series of total property value estimates indexed by locations.
        """
        total_values = []

        def format_sector(sect):
            s = sect.split()
            return s[0] + ' ' + s[1]

        self.household_data = self.household_data.reset_index()
        self.household_data['postcode sector'] = self.household_data['postcode sector'].apply(lambda x: format_sector(x))
        
        self.household_data.set_index('postcode sector', inplace=True)

        for p_code in postal_data:

            sp = p_code.split()
            if len(sp[1]) > 1:

                unitprice = self.get_median_house_price_estimate([p_code])

                total_values.append(float(unitprice))
            
            else:
                num_houses = self.household_data.loc[p_code]['households']
                units = self.postcodedb.loc[self.postcodedb['sector'] == p_code]
                postcodes = list(units['postcode'])
                prices = self.get_median_house_price_estimate(postcodes)
                avg_price = prices.mean()

                total_values.append(avg_price * num_houses)

        return pd.Series(total_values, index=postal_data)


    def get_annual_flood_risk(self, postcodes,  risk_labels=None):
        """
        Return a series of estimates of the total property values of a
        collection of postcodes.

        Risk is defined here as a damage coefficient multiplied by the
        value under threat multiplied by the probability of an event.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        risk_labels: pandas.Series (optional)
            Series containing flood risk classifiers, as
            predicted by get_flood_class_from_postcodes.

        Returns
        -------

        pandas.Series
            Series of total annual flood risk estimates indexed by locations.
        """

        if not risk_labels:
            risk_labels = self.get_flood_class_from_postcodes(postcodes)

        risk_label_to_percent = {1:0.00001, 2:0.0005, 3:0.001, 4:0.005, 5:0.01, 6:0.015, 7:0.02, 8:0.03, 9:0.04, 10:0.05}

        percent_risk = risk_labels.replace(risk_label_to_percent)

        cost = self.get_total_value(risk_labels.index)

        return percent_risk * cost


    def get_predicitions_for_unlabelled_data(self, data):
        """
        Return a series of estimates of the flood class and median house price of a
        collection of postcodes.

        Parameters
        ----------

        data : pandas.DataFrame 
            containing (postcode, sector, longitude, latitude, localAuthority, soilType, altitude)

        Returns
        -------

        pandas.DataFrame
            The original data with the predicted 
        """

        floodclass_preds = self.floodClassModel.predict(data)
        medianprice_preds = self.medianPriceModel.predict(self.median_price_model_trained, data)

        floodclass_preds = pd.Series(floodclass_preds)
        medianprice_preds = pd.Series(medianprice_preds)
        results = data.copy()
        results['riskLabel'] = floodclass_preds
        results['medianPrice'] = medianprice_preds
        return results