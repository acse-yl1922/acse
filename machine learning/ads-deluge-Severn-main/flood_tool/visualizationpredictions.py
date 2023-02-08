from .tool import Tool
import pandas as pd
from .geo import *
import numpy as np


class VisualizationPredictions():

    def __init__(self, filepath='flood_tool/resources/postcodes_unlabelled.csv'):
        self.data = pd.read_csv(filepath)
        self.t = Tool()
        self.t.train()

    def get_predictions(self):
        easting, northing = np.array(self.data['easting']), np.array(self.data['northing'])
        lat, long = get_gps_lat_long_from_easting_northing(easting, northing)

        self.data['latitude'] = lat
        self.data['longitude'] = long
        self.data = self.data.drop(columns=['easting', 'northing'])
        
        data = self.t.get_predicitions_for_unlabelled_data(self.data)
        print(data)

        return data